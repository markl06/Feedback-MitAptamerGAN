from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import re
import subprocess

import torch

from .data import normalize_sequence


class ErnieRNAEncoder(ABC):
    embedding_dim: int

    @abstractmethod
    def encode(self, sequences: list[str]) -> torch.Tensor:
        """Return finite [batch, embedding_dim] float features, excluding padding."""
        raise NotImplementedError


@dataclass(frozen=True)
class FoldResult:
    structure: str
    mfe: float


class RNAfoldBackend(ABC):
    @abstractmethod
    def fold(self, sequences: list[str]) -> list[FoldResult]:
        raise NotImplementedError


class RNAfoldCLI(RNAfoldBackend):
    """Optional concrete adapter; only invoked when explicitly configured.

    Uses subprocess argument lists, no shell. Temperature is explicit; other
    thermodynamic options follow the installed RNAfold version's defaults.
    """
    def __init__(self, executable="RNAfold", temperature=37.0, timeout=60):
        self.executable = executable
        self.temperature = float(temperature)
        self.timeout = float(timeout)

    def fold(self, sequences):
        results = []
        for sequence in sequences:
            normalize_sequence(sequence)
            process = subprocess.run(
                [self.executable, "--noPS", f"--temp={self.temperature}"],
                input=sequence + "\n", text=True, encoding="utf-8", capture_output=True,
                check=True, timeout=self.timeout,
            )
            matches = re.findall(r"^([.()]+)\s+\(\s*([-+0-9.eE]+)\s*\)\s*$",
                                 process.stdout, flags=re.MULTILINE)
            if len(matches) != 1:
                raise ValueError(f"Cannot parse RNAfold MFE output: {process.stdout!r}")
            structure, energy = matches[0]
            results.append(FoldResult(structure, float(energy)))
        return results


def load_factory(spec, kwargs=None):
    """Load a trusted local adapter via 'package.module:factory'."""
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("Adapter must have form package.module:factory")
    return getattr(importlib.import_module(module), name)(**(kwargs or {}))


def read_backend_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if set(config) - {"ernie", "rnafold"}:
        raise ValueError("Unknown backend configuration key")
    return config


class FeatureExtractor:
    """Freeze external features; feature definition is an explicit assumption.

    Fold vector: 70 x 3 one-hot dot/open/close (padding all zeros), MFE,
    MFE/nucleotide, paired nucleotide fraction, length/70, maximum depth/70.
    Includes structure and thermodynamics, without claiming the unpublished head.
    """
    def __init__(self, config, backends):
        self.config = config
        self.specification = backends
        self.ernie, self.rnafold = None, None
        if config.use_ernie:
            entry = backends.get("ernie")
            if not entry:
                raise ValueError("ERNIE-RNA adapter missing; configure it or explicitly disable the branch")
            self.ernie = load_factory(entry["factory"], entry.get("kwargs"))
            if self.ernie.embedding_dim != config.embedding_dim:
                raise ValueError("ERNIE-RNA embedding_dim differs from model configuration")
        if config.use_rnafold:
            entry = backends.get("rnafold")
            if not entry:
                raise ValueError("RNAfold adapter missing; configure it or explicitly disable the branch")
            self.rnafold = load_factory(entry["factory"], entry.get("kwargs"))

    @property
    def output_dim(self):
        return ((self.config.embedding_dim if self.config.use_ernie else 0)
                + (self.config.max_length * 3 + 5 if self.config.use_rnafold else 0))

    def fold_features(self, sequences, results):
        if len(sequences) != len(results):
            raise ValueError("RNAfold returned wrong batch size")
        output = []
        for sequence, result in zip(sequences, results):
            structure = result.structure
            if len(structure) != len(sequence) or set(structure) - set(".()"):
                raise ValueError("Invalid dot-bracket structure")
            if not math.isfinite(result.mfe):
                raise ValueError("MFE must be finite")
            encoding = torch.zeros(self.config.max_length, 3)
            depth, max_depth = 0, 0
            for i, symbol in enumerate(structure):
                encoding[i, ".()".index(symbol)] = 1
                depth += int(symbol == "(") - int(symbol == ")")
                max_depth = max(max_depth, depth)
                if depth < 0:
                    raise ValueError("Unbalanced dot-bracket structure")
            if depth:
                raise ValueError("Unbalanced dot-bracket structure")
            length = len(sequence)
            summary = torch.tensor([result.mfe, result.mfe / length,
                                    (structure.count("(") + structure.count(")")) / length,
                                    length / self.config.max_length, max_depth / self.config.max_length])
            output.append(torch.cat([encoding.flatten(), summary]))
        return torch.stack(output)

    @torch.no_grad()
    def extract(self, sequences, batch_size=64):
        if not sequences:
            return torch.empty(0, self.output_dim)
        batches = []
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            for sequence in batch:
                if normalize_sequence(sequence, max_length=self.config.max_length) != sequence:
                    raise ValueError("External adapters require canonical RNA sequences")
            features = []
            if self.ernie is not None:
                embedded = torch.as_tensor(self.ernie.encode(batch)).detach().to(device="cpu", dtype=torch.float32)
                if embedded.shape != (len(batch), self.config.embedding_dim):
                    raise ValueError("ERNIE-RNA must return [B, D] pooled features")
                features.append(embedded)
            if self.rnafold is not None:
                features.append(self.fold_features(batch, self.rnafold.fold(batch)))
            combined = torch.cat(features, dim=-1)
            if not torch.isfinite(combined).all():
                raise ValueError("Non-finite evaluator features")
            batches.append(combined)
        return torch.cat(batches)
