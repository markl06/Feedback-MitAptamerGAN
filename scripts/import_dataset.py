"""Normalize CSV/TSV/FASTA/JSONL sources without importing ML dependencies.

Run from project root: python -m scripts.import_dataset --help
No primer removal, truncation, ambiguous-base imputation, or negative synthesis.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random

from mitaptamer.data import normalize_sequence, sequence_id, write_jsonl


def iter_records(path, sequence_column, id_column, group_column):
    suffix = path.suffix.lower()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        if suffix in {".fa", ".fasta", ".fna"}:
            name, parts = None, []
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if name is not None:
                        yield {sequence_column: "".join(parts), id_column: name}
                    name, parts = line[1:].strip(), []
                else:
                    if name is None:
                        raise ValueError(f"{path}:{number}: FASTA requires a header")
                    parts.append(line)
            if name is not None:
                yield {sequence_column: "".join(parts), id_column: name}
        elif suffix in {".csv", ".tsv"}:
            reader = csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ",")
            if sequence_column not in (reader.fieldnames or []):
                raise ValueError(f"{path}: missing column {sequence_column}")
            yield from reader
        elif suffix == ".jsonl":
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}: each JSONL row must be an object")
                    yield row
        else:
            raise ValueError(f"Unsupported input format: {path}")


def assign_splits(rows, ratios, seed):
    """Keep exact duplicates and user-supplied families in one component.

    Greedy group-stratified assignment is approximate when family sizes vary.
    All components touching any reference are forced into training.
    """
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    group_owner = {}
    for i, row in enumerate(rows):
        for group in row["groups"]:
            if group in group_owner:
                parent[find(i)] = find(group_owner[group])
            else:
                group_owner[group] = i
    components = {}
    for i in range(len(rows)):
        components.setdefault(find(i), []).append(i)
    names = ("train", "validation", "test")
    totals = Counter(row["label"] for row in rows)
    target = {name: {label: ratios[k] * count for label, count in totals.items()}
              for k, name in enumerate(names)}
    counts = {name: Counter() for name in names}

    def put(indices, split):
        for i in indices:
            rows[i]["split"] = split
            counts[split][rows[i]["label"]] += 1

    remaining = []
    for indices in components.values():
        if any(rows[i]["source"] == "reference" for i in indices):
            put(indices, "train")
        else:
            remaining.append(indices)
    rng = random.Random(seed)
    rng.shuffle(remaining)
    remaining.sort(key=len, reverse=True)
    for indices in remaining:
        added = Counter(rows[i]["label"] for i in indices)
        # Increment in squared distance to all class-specific split targets.
        def cost(name):
            return sum(((counts[name][label] + n - target[name][label]) ** 2
                        - (counts[name][label] - target[name][label]) ** 2)
                       / max(target[name][label], 1) for label, n in added.items())
        put(indices, min(names, key=cost))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--selex", type=Path, nargs="+", required=True)
    parser.add_argument("--background", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--group-column", default="group")
    parser.add_argument("--require-groups", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=70, choices=[70])
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    parser.add_argument("--reference-mass", type=float, default=0.15)
    parser.add_argument("--positive-class-mass", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(r <= 0 for r in args.ratios) or abs(sum(args.ratios) - 1) > 1e-8:
        parser.error("--ratios must be positive and sum to 1")
    if not 1 <= args.min_length <= args.max_length:
        parser.error("Invalid length bounds")
    if not 0 < args.reference_mass < 1 or not 0 < args.positive_class_mass < 1:
        parser.error("Source and class masses must be in (0, 1)")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new path")
    merged, rejected, files = {}, [], []
    raw_counts = Counter()
    for source in ("reference", "selex", "background"):
        for path in getattr(args, source):
            files.append({"path": str(path.resolve()), "source": source,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            for index, record in enumerate(iter_records(path, args.sequence_column,
                                                        args.id_column, args.group_column), 1):
                raw_counts[source] += 1
                try:
                    sequence = normalize_sequence(record.get(args.sequence_column),
                                                  args.min_length, args.max_length)
                    group = str(record.get(args.group_column) or "").strip()
                    if args.require_groups and not group:
                        raise ValueError("Missing sequence family/cluster group")
                except ValueError as error:
                    rejected.append({"file": str(path), "record": index, "reason": str(error)})
                    continue
                label = int(source != "background")
                supplied_label = record.get("label")
                if supplied_label not in (None, "") and str(supplied_label).strip() != str(label):
                    raise ValueError(f"{path}:{index}: label conflicts with declared source {source}")
                # Conflicting positive/negative annotations always abort.
                if sequence in merged and merged[sequence]["label"] != label:
                    raise ValueError(f"Conflicting source labels for {sequence_id(sequence)} at {path}:{index}")
                if sequence not in merged:
                    merged[sequence] = {"id": sequence_id(sequence), "sequence": sequence,
                                        "length": len(sequence), "label": label, "source": source,
                                        "sources": [], "groups": [], "provenance": []}
                row = merged[sequence]
                if source == "reference":
                    row["source"] = "reference"
                if source not in row["sources"]:
                    row["sources"].append(source)
                if group and group not in row["groups"]:
                    row["groups"].append(group)
                row["provenance"].append({"file": str(path.resolve()), "record": index,
                                          "original_id": record.get(args.id_column), "source": source,
                                          "metadata": {k: v for k, v in record.items()
                                                       if k != args.sequence_column}})
    if rejected and not args.skip_invalid:
        raise ValueError(f"{len(rejected)} invalid records. First errors: {rejected[:10]}. "
                         "Fix input or explicitly use --skip-invalid.")
    rows = sorted(merged.values(), key=lambda row: row["id"])
    assign_splits(rows, args.ratios, args.seed)
    split_rows = {name: [r for r in rows if r["split"] == name]
                  for name in ("train", "validation", "test")}
    for name, subset in split_rows.items():
        if {r["label"] for r in subset} != {0, 1}:
            raise ValueError(f"{name} lacks both labels; supply more independent sequence groups")
    from mitaptamer.data import source_weights
    train = split_rows["train"]
    weights = source_weights(train, args.reference_mass, args.positive_class_mass)
    for row, weight in zip(train, weights):
        row["evaluator_sampling_weight"] = weight
    positives = [r for r in train if r["label"] == 1]
    for row, weight in zip(positives, source_weights(positives, args.reference_mass, positive_only=True)):
        row["gan_sampling_weight"] = weight
    warnings = []
    if any(not row["groups"] for row in rows):
        warnings.append("Some sequences lack family groups: only exact-sequence leakage is prevented for these rows.")
    if sum(r["source"] == "reference" for r in rows) != 3:
        warnings.append("Unique reference count differs from the paper's three references.")
    if sum("selex" in r["sources"] for r in rows) != 20000:
        warnings.append("Unique SELEX count differs from the paper's 20,000 sequences.")
    report = {"schema_version": 1, "seed": args.seed, "alphabet": "ACGU", "max_length": 70,
              "normalization": "strip whitespace; uppercase; T->U; reject all other symbols; no truncation",
              "requested_ratios": args.ratios, "reference_mass": args.reference_mass,
              "positive_class_mass": args.positive_class_mass, "files": files,
              "input_counts": dict(raw_counts), "unique_count": len(rows),
              "duplicate_records_merged": sum(raw_counts.values()) - len(rejected) - len(rows),
              "rejected_count": len(rejected), "warnings": warnings,
              "splits": {name: {"count": len(subset), "labels": dict(Counter(r["label"] for r in subset)),
                                "sources": dict(Counter(r["source"] for r in subset))}
                         for name, subset in split_rows.items()}}
    args.output.mkdir(parents=True)
    for name, subset in split_rows.items():
        write_jsonl(args.output / f"{name}.jsonl", subset)
    write_jsonl(args.output / "rejected.jsonl", rejected)
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
