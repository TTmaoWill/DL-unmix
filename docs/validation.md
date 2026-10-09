# Validation coverage

These checks concern software behavior and numerical conversion. They do not
establish biological accuracy, downstream association validity or a new
scientific benchmark result.

## Package 0.3.0

The implementation and tests at commit
`b370484b22ce5d45bf2a542fb29f972226dc6d53` were checked in the supported
Python 3.11.14, NumPy 2.3.5 and pandas 2.3.3 environment:

| Environment | Executed result |
| --- | --- |
| PyTorch 2.10.0+cpu | 14 passed; 2 conditional CUDA tests skipped |
| PyTorch 2.10.0+cu126, Tesla V100-SXM2-16GB | 16 passed; no skips |

Coverage includes fit/save/load/predict, label checks, failed-refit state
preservation, CLI roundtrips, explicit device selection and artifact conversion.
Synthetic parity checks use 2, 3 and 5 cell types, both fraction floors, and
aligned initial weights and random states. Full epoch-selection/refit
comparisons ran on CPU under both PyTorch builds; mapped three-epoch training
and the regular CUDA fitting tests also ran on CUDA. CPU/CUDA training is not
claimed to be bitwise identical. The numerical tolerance is
`atol=2e-5, rtol=1e-5`.

Run the suite with:

```bash
python -m unittest discover -s release_tests -v
```

The external reference parity test requires the source described in
[provenance](provenance.md); without it, the test is explicitly skipped.
CUDA tests similarly require an available allocated GPU. See
[installation](installation.md) for platform coverage.

## ROSMAP artifact conversion

A separate private artifact check restored the original reference metadata,
converted the fitted model, reloaded both formats and compared all
1,037 target samples × 15,131 genes × 5 cell types: **78,454,235 values**.
The original seed (20260307), selected epoch count (10), reference cohort
(341 training + 86 validation = 427), and prediction floor (1e-6) were retained.
No training was performed.

| Comparison on processed expression scale | Mean absolute difference | Maximum absolute difference | Outside tolerance |
| --- | ---: | ---: | ---: |
| Converted vs original implementation | 0 | 0 | 0 |
| Either implementation vs historical TSV | 3.398276e-7 | 2.448242e-4 | 0 |

Weights were checked against the original checkpoint after the documented
column mapping. Reference features, scalers, restored validation PCC and
selection masks were preserved through conversion. Restored validation
medians and finite-gene counts matched the archived validation summaries.

This full-cohort check ran on macOS CPU with pre-existing Python 3.9.13,
PyTorch 2.4.1, NumPy 2.0.2 and pandas 2.2.2. It is numerical compatibility
evidence in that environment, **not verification of the supported release
environment or an endorsement of those older versions**. The declared
Python/PyTorch requirements remain unchanged. ROSMAP GPU execution was not
tested in this check. Individual predictions and donor identifiers are not
included in this repository.
