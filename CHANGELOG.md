# Version notes

## 0.3.0 — 2026-10-09

- Compact reference-supervised residual MLP with CPU and explicit CUDA execution.
- Portable format-2 models containing configuration, reference features,
  validation PCC and tensor weights.
- One-time conversion of trusted version-0.2 artifacts with
  `tools/convert_model.py`; normal loading accepts format 2.
- Python API, CLI and synthetic workflow for fitting and prediction.

The default fitting configuration uses seed 20260307; validation checks at
epochs 5, 10, 15 and 20; batch sizes 4096/8192 for training/evaluation;
AdamW with learning rate 0.0003 and weight decay 0.0001; and dropout 0.15.
The selected epoch count is used for a fresh full-reference refit.

Training uses a fraction floor of 1e-6. Prediction defaults to 0.01 and accepts
an explicit `fraction_floor`; preserving a prior analysis requires preserving
that analysis's setting. Expression input is normalized linear expression;
predictions are on the log2(max(expression, 0) + 1) scale. See the
[method](docs/method.md) and [validation record](docs/validation.md).

This is a package release. No pretrained model files are distributed.
