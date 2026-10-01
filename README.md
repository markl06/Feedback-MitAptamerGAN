# Feedback-MitAptamerGAN
For SELEX-informed generation and prioritization of mitochondrial RNA aptamer candidates

## Files

```text
mitaptamer/
  config.py               Paper parameters and implementation assumptions
  data.py                 Canonical data loading, encoding, decoding, and source weights
  models.py               Generator, Wasserstein critic, evaluator, and gradient penalty
  integrations.py         ERNIE-RNA / RNAfold interfaces and optional RNAfold CLI adapter
  training.py             Classifier training, WGAN-GP, feedback, evaluation, and generation
adapters/
  ernie_rna.py             Adapter stub for a local ERNIE-RNA implementation
  rnafold.py               Adapter stub for alternative RNAfold implementations
scripts/
  import_dataset.py       Standalone data normalization using only the Python standard library
  run.py                  Training, evaluation, and generation entry points
  make_search_configs.py  Write the paper's parameter grid without launching training
configs/
  default.json            Default model configuration
  backends.example.json   Example external backend configuration
```
