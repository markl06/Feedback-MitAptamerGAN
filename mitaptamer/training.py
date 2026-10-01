from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .config import Config
from .data import (decode_tokens, encode_sequences, load_dataset, sequence_id,
                   source_weights, write_jsonl)
from .integrations import FeatureExtractor
from .models import Critic, Evaluator, Generator, gradient_penalty


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Exact reproducibility across GPU/PyTorch/adapter versions is not promised.


def dataset_fingerprint(directory):
    digest = hashlib.sha256()
    for split in ("train", "validation", "test"):
        digest.update(split.encode())
        digest.update((Path(directory) / f"{split}.jsonl").read_bytes())
    return digest.hexdigest()


def file_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_run(path):
    path = Path(path)
    # Avoid accidentally overwriting selected models or prior experiment logs.
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predicted = probabilities >= 0.5
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "auroc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None}


@torch.no_grad()
def scores_for_features(model, features, device, batch_size):
    model.eval()
    if not len(features):
        return torch.empty(0)
    return torch.cat([model.score(batch.to(device)).cpu()
                      for batch in features.split(batch_size)])


def train_evaluator(dataset_dir, output_dir, config, backends, device="cpu"):
    config.validate()
    seed_everything(config.seed)
    data = load_dataset(dataset_dir)
    extractor = FeatureExtractor(config, backends)
    train_rows, validation_rows = data["train"], data["validation"]
    # No test features/scores are used for optimizer or checkpoint selection.
    train_x = extractor.extract([r["sequence"] for r in train_rows], config.batch_size)
    validation_x = extractor.extract([r["sequence"] for r in validation_rows], config.batch_size)
    train_y = torch.tensor([r["label"] for r in train_rows], dtype=torch.float32)
    validation_y = torch.tensor([r["label"] for r in validation_rows], dtype=torch.float32)
    weights = torch.tensor(source_weights(train_rows, config.reference_mass, config.positive_class_mass),
                           dtype=torch.float64)
    model = Evaluator(extractor.output_dim, config).to(device)
    model.fit_normalization(train_x.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.evaluator_lr, betas=config.evaluator_betas)
    run_dir = prepare_run(output_dir)
    fingerprint = dataset_fingerprint(dataset_dir)
    save_json(run_dir / "config.json", config.to_dict())
    best_loss = float("inf")
    history = []
    for epoch in range(config.evaluator_epochs):
        model.train()
        train_losses = []
        for _ in range(math.ceil(len(train_rows) / config.batch_size)):
            indices = torch.multinomial(weights, config.batch_size, replacement=True)
            logits = model(train_x[indices].to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y[indices].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluator loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_logits = torch.cat([model(x.to(device)).cpu() for x in validation_x.split(config.batch_size)])
            validation_loss = nn.functional.binary_cross_entropy_with_logits(val_logits, validation_y).item()
        if not math.isfinite(validation_loss):
            raise FloatingPointError("Non-finite validation loss")
        summary = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)),
                   "validation_loss": validation_loss,
                   "validation_metrics": metrics(validation_y.numpy(), val_logits.sigmoid().numpy())}
        history.append(summary)
        print(json.dumps(summary))
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save({"kind": "evaluator", "config": config.to_dict(), "backends": backends,
                        "input_dim": extractor.output_dim, "model": model.state_dict(),
                        "epoch": epoch + 1, "validation_loss": validation_loss,
                        "dataset_fingerprint": fingerprint}, run_dir / "evaluator.pt")
        save_json(run_dir / "history.json", history)
    return run_dir / "evaluator.pt"


def load_evaluator(checkpoint, backends=None, device="cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("kind") != "evaluator":
        raise ValueError("Expected evaluator checkpoint")
    config = Config(**saved["config"]).validate()
    effective_backends = saved["backends"] if backends is None else backends
    if effective_backends != saved["backends"]:
        raise ValueError("Backend specifications differ from training; use identical model/settings")
    extractor = FeatureExtractor(config, effective_backends)
    if saved["input_dim"] != extractor.output_dim:
        raise ValueError("Evaluator feature dimensions changed")
    model = Evaluator(extractor.output_dim, config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    model.requires_grad_(False)
    return model, extractor, saved


def evaluate(dataset_dir, checkpoint, output, device="cpu", split="test"):
    data = load_dataset(dataset_dir)
    model, extractor, saved = load_evaluator(checkpoint, device=device)
    if dataset_fingerprint(dataset_dir) != saved["dataset_fingerprint"]:
        raise ValueError("Dataset differs from the evaluator's training split manifest")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = data[split]
    features = extractor.extract([r["sequence"] for r in rows], extractor.config.batch_size)
    scores = scores_for_features(model, features, device, extractor.config.batch_size).tolist()
    result = {"split": split, "checkpoint_sha256": file_fingerprint(checkpoint),
              "metrics": metrics([r["label"] for r in rows], scores),
              "predictions": [{"id": r["id"], "label": r["label"], "score": s}
                              for r, s in zip(rows, scores)]}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)


@torch.no_grad()
def propose(generator, count, device, batch_size, min_length):
    previous_mode = generator.training
    generator.eval()
    result = []
    try:
        for start in range(0, count, batch_size):
            size = min(batch_size, count - start)
            probabilities = generator(torch.randn(size, generator.latent_dim, device=device))
            # Sample independent position categories, stopping at the first Z.
            tokens = torch.multinomial(probabilities.reshape(-1, 5), 1).reshape(size, generator.length)
            result.extend(seq for seq in decode_tokens(tokens.cpu().tolist()) if len(seq) >= min_length)
    finally:
        generator.train(previous_mode)
    return result


def mutate_proposals(sequences, fraction, rng):
    result = list(sequences)
    selected = rng.sample(range(len(result)), round(len(result) * fraction))
    for index in selected:
        sequence = result[index]
        position = rng.randrange(len(sequence))
        alternatives = [base for base in "ACGU" if base != sequence[position]]
        result[index] = sequence[:position] + rng.choice(alternatives) + sequence[position + 1:]
    return result


@dataclass
class FeedbackPool:
    """Fixed nonreference slots; references remain a separate 15% sampling mass.

    Default feedback_fraction=0.75 caps synthetic occupancy at 75% of the
    nonreference pool. Highest-scoring accumulated candidates replace its prefix.
    Rand-Syn uses uniform sampling and no reference weighting because its initial
    pool is random. Duplicate base slots are allowed as empirical resampling.
    """
    references: list[str]
    baseline: list[str]
    synthetic: dict[str, float]
    config: Config

    @classmethod
    def build(cls, rows, config, rng, forbidden):
        references = [r["sequence"] for r in rows if r["source"] == "reference"]
        enriched = [r["sequence"] for r in rows if r["source"] == "selex"]
        if not references or not enriched:
            raise ValueError("GAN training requires reference and SELEX sequences")
        size = config.feedback_pool_size
        if config.strategy == "Pos-PosSyn":
            # Preserve each empirical SELEX row; feedback_pool_size applies to random strategies.
            baseline = list(enriched)
            rng.shuffle(baseline)
        else:
            n_random = size if config.strategy == "Rand-Syn" else round(size * config.random_start_fraction)
            baseline = rng.choices(enriched, k=size - n_random)
            # Length-matched random initialization is an implementation assumption.
            lengths = [len(sequence) for sequence in enriched]
            max_attempts = max(1000, 100 * n_random)
            for _ in range(max_attempts):
                if len(baseline) >= size:
                    break
                sequence = "".join(rng.choices("ACGU", k=rng.choice(lengths)))
                if sequence not in forbidden:
                    baseline.append(sequence)
            if len(baseline) < size:
                raise ValueError("Cannot construct random pool without holdout/negative collisions")
            rng.shuffle(baseline)
        return cls(references, baseline, {}, config)

    def sequences_and_weights(self):
        selected = sorted(self.synthetic, key=lambda seq: (-self.synthetic[seq], seq))
        nonreference = selected + self.baseline[len(selected):]
        if self.config.strategy == "Rand-Syn":
            return nonreference, [1 / len(nonreference)] * len(nonreference)
        ref_mass = self.config.reference_mass
        return (self.references + nonreference,
                [ref_mass / len(self.references)] * len(self.references)
                + [(1 - ref_mass) / len(nonreference)] * len(nonreference))

    def update(self, scored):
        capacity = int(len(self.baseline) * self.config.feedback_fraction)
        combined = dict(self.synthetic)
        for seq, score in scored:
            if score > self.config.feedback_threshold:
                combined[seq] = max(score, combined.get(seq, -1))
        ordered = sorted(combined, key=lambda seq: (-combined[seq], seq))[:capacity]
        self.synthetic = {seq: combined[seq] for seq in ordered}


def train_gan(dataset_dir, output_dir, config, evaluator_checkpoint=None, device="cpu"):
    config.validate()
    seed_everything(config.seed)
    rng = random.Random(config.seed)
    data = load_dataset(dataset_dir)
    fingerprint = dataset_fingerprint(dataset_dir)
    evaluator, extractor, evaluator_hash = None, None, None
    if config.use_feedback:
        if evaluator_checkpoint is None:
            raise ValueError("Feedback training requires --evaluator")
        evaluator, extractor, saved = load_evaluator(evaluator_checkpoint, device=device)
        if fingerprint != saved["dataset_fingerprint"]:
            raise ValueError("GAN and evaluator must use identical data partitions")
        if (config.use_ernie, config.use_rnafold) != (extractor.config.use_ernie, extractor.config.use_rnafold):
            raise ValueError("GAN/evaluator feature ablations differ")
        evaluator_hash = file_fingerprint(evaluator_checkpoint)
    all_originals = {row["sequence"] for rows in data.values() for row in rows}
    pool = FeedbackPool.build(data["train"], config, rng, all_originals)
    generator, critic = Generator(config).to(device), Critic(config).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.gan_lr, betas=config.gan_betas)
    d_optimizer = torch.optim.Adam(critic.parameters(), lr=config.gan_lr, betas=config.gan_betas)
    run_dir = prepare_run(output_dir)
    save_json(run_dir / "config.json", config.to_dict())
    history, feedback_history = [], []
    first_update = math.ceil(config.update_point * config.gan_epochs)

    def update_feedback(completed_epochs):
        proposals = propose(generator, config.feedback_candidates, device, config.batch_size, config.min_length)
        proposals = mutate_proposals(proposals, config.mutation_fraction, rng)
        unique = sorted(set(proposals) - all_originals)
        if unique:
            features = extractor.extract(unique, config.batch_size)
            scores = scores_for_features(evaluator, features, device, config.batch_size).tolist()
            scored = list(zip(unique, scores))
        else:
            scored = []
        pool.update(scored)
        accepted = [(seq, score) for seq, score in scored if score > config.feedback_threshold]
        # Auditable computational labels, never appended to supervised source data.
        write_jsonl(run_dir / f"feedback_{completed_epochs:04d}.jsonl",
                    ({"id": sequence_id(seq), "sequence": seq, "score": score,
                      "source": "synthetic_feedback", "selected": score > config.feedback_threshold,
                      "in_pool": seq in pool.synthetic} for seq, score in scored))
        feedback_history.append({"completed_epochs": completed_epochs, "unique_scored": len(scored),
                                 "above_threshold": len(accepted), "synthetic_pool_size": len(pool.synthetic)})

    for epoch in range(config.gan_epochs):
        if (config.use_feedback and epoch >= first_update
                and (epoch - first_update) % config.feedback_every == 0):
            update_feedback(epoch)
        sequences, weights = pool.sequences_and_weights()
        real_encodings = encode_sequences(sequences, config.max_length)
        sampling_weights = torch.tensor(weights, dtype=torch.float64)
        generator.train()
        critic.train()
        d_values, g_values, gp_values = [], [], []
        for _ in range(config.steps_per_epoch):
            critic.requires_grad_(True)
            for _ in range(config.critic_steps):
                indices = torch.multinomial(sampling_weights, config.batch_size, replacement=True)
                real = real_encodings[indices].to(device)
                with torch.no_grad():
                    fake = generator(torch.randn(config.batch_size, config.latent_dim, device=device))
                gp = gradient_penalty(critic, real, fake)
                d_loss = critic(fake).mean() - critic(real).mean() + config.gp_lambda * gp
                if not torch.isfinite(d_loss):
                    raise FloatingPointError("Non-finite critic loss")
                d_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                d_optimizer.step()
                d_values.append(d_loss.item())
                gp_values.append(gp.item())
            critic.requires_grad_(False)
            fake = generator(torch.randn(config.batch_size, config.latent_dim, device=device))
            g_loss = -critic(fake).mean()
            if not torch.isfinite(g_loss):
                raise FloatingPointError("Non-finite generator loss")
            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()
            g_values.append(g_loss.item())
        summary = {"epoch": epoch + 1, "critic_loss": float(np.mean(d_values)),
                   "generator_loss": float(np.mean(g_values)), "gradient_penalty": float(np.mean(gp_values)),
                   "synthetic_pool_size": len(pool.synthetic)}
        print(json.dumps(summary))
        history.append(summary)
        torch.save({"kind": "gan", "config": config.to_dict(), "generator": generator.state_dict(),
                    "critic": critic.state_dict(), "epoch": epoch + 1,
                    "dataset_fingerprint": fingerprint, "evaluator_sha256": evaluator_hash}, run_dir / "gan.pt")
        save_json(run_dir / "history.json", {"training": history, "feedback": feedback_history})
    if config.use_feedback and first_update == config.gan_epochs:
        update_feedback(config.gan_epochs)
        save_json(run_dir / "history.json", {"training": history, "feedback": feedback_history})
    sequences, weights = pool.sequences_and_weights()
    write_jsonl(run_dir / "final_pool.jsonl", (
        {"sequence": seq, "sampling_weight": weight, "synthetic": seq in pool.synthetic,
         "score": pool.synthetic.get(seq)} for seq, weight in zip(sequences, weights)))
    return run_dir / "gan.pt"


def generate(dataset_dir, checkpoint, output, evaluator_checkpoint=None, count=205,
             proposals=10000, rounds=10, seed=42, device="cpu", threshold=None):
    if count < 1 or proposals < 1 or rounds < 1:
        raise ValueError("count/proposals/rounds must be positive")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if threshold is not None and evaluator_checkpoint is None:
        raise ValueError("A threshold requires an evaluator")
    seed_everything(seed)
    data = load_dataset(dataset_dir)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("kind") != "gan":
        raise ValueError("Expected GAN checkpoint")
    fingerprint = dataset_fingerprint(dataset_dir)
    if fingerprint != saved["dataset_fingerprint"]:
        raise ValueError("Generation dataset differs from GAN training dataset")
    config = Config(**saved["config"]).validate()
    generator = Generator(config).to(device)
    generator.load_state_dict(saved["generator"])
    generator.eval()
    evaluator, extractor = None, None
    if evaluator_checkpoint is not None:
        evaluator, extractor, evaluation_saved = load_evaluator(evaluator_checkpoint, device=device)
        if fingerprint != evaluation_saved["dataset_fingerprint"]:
            raise ValueError("Evaluator data partitions differ")
        expected_hash = saved.get("evaluator_sha256")
        if expected_hash is not None and file_fingerprint(evaluator_checkpoint) != expected_hash:
            raise ValueError("Evaluator differs from the feedback checkpoint")
    originals = {row["sequence"] for rows in data.values() for row in rows}
    visited, records = set(), []
    # Bounded rejection sampling; report shortfalls rather than silently relax filters.
    for _ in range(rounds):
        candidates = sorted(set(propose(generator, proposals, device, config.batch_size, config.min_length))
                            - originals - visited)
        visited.update(candidates)
        if not candidates:
            continue
        scores = ([None] * len(candidates) if evaluator is None else scores_for_features(
            evaluator, extractor.extract(candidates, config.batch_size), device, config.batch_size).tolist())
        records.extend({"id": sequence_id(seq), "sequence": seq, "length": len(seq), "score": score,
                        "source": "generated", "exact_novelty": True}
                       for seq, score in zip(candidates, scores) if threshold is None or score > threshold)
        if len(records) >= count:
            break
    records.sort(key=lambda row: (-(row["score"] if row["score"] is not None else 0), row["id"]))
    run_dir = prepare_run(output)
    write_jsonl(run_dir / "candidates.jsonl", records[:count])
    with (run_dir / "candidates.fasta").open("w", encoding="utf-8") as handle:
        for row in records[:count]:
            handle.write(f">{row['id']} score={row['score']}\n{row['sequence']}\n")
    save_json(run_dir / "manifest.json", {
        "requested": count, "returned": min(count, len(records)), "shortfall": max(0, count - len(records)),
        "unique_novel_candidates": len(visited),
        "unique_novel_candidates_scored": len(visited) if evaluator is not None else 0,
        "threshold_strictly_greater_than": threshold,
        "gan_sha256": file_fingerprint(checkpoint), "seed": seed,
        "evaluator_sha256": file_fingerprint(evaluator_checkpoint) if evaluator_checkpoint else None,
        "dataset_fingerprint": fingerprint, "novelty_exclusion": "all three original splits",
        "note": "Scores predict source labels, not measured binding affinity. Exact novelty is not family novelty."})
