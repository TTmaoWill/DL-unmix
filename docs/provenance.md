# Source provenance and verification

The adopted implementation is the `DL_source` snapshot accompanying the
`fujita_mathys_no_overlap_all_methods_20261005` rerun, referenced by the adopted
`figure2_disjoint_reference_20261008` source manifest. The latter records audit
and result hashes, not a historical hash of the Python source itself. This
release records the inspected source hashes in `provenance.json`; it does not
assert an independently established historical byte identity.

`dlunmix/_reference.py` is copied from `dl_unmix_common.py`, with one bounded
change: `build_bundle` accepts absent CTS truth and returns `Y_abs=None` and
`truth_raw_df=None` for inference. With truth supplied, original calculations
are preserved. Model defaults are fixed by the public wrapper to the adopted
variant, rather than the shared library's broader historical defaults.

The original train/predict scripts are not installed or included as alternate
entrypoints. Their hashes are recorded for provenance. No reference donors,
clinical variables, identity crosswalks, research predictions, real genotypes,
research model weights or large generated figures are copied into this release.

## Software verification

The regular unittest suite covers fit/save/load/predict, deterministic seeded
training, label alignment, invalid inputs, fraction floors, output selection,
optional evaluation and CLI file roundtrips. It generates all data at runtime.

To run the additional reference parity check, provide a local directory
containing the original `dl_unmix_common.py`, `train_step1.py` and
`train_step2.py` matching the recorded hashes:

```bash
DLUNMIX_REFERENCE_SOURCE=/path/to/adopted/DL_source \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s release_tests -v
```

That test uses a temporary directory and generated synthetic data, invokes the
original scripts for two candidate epochs, and compares features/scalers,
selected epoch, full-refit weights and prediction at both fraction floors. It
does not read original research data or pretrained weights. The source directory
is never modified. Without the environment variable the external-reference
test is explicitly skipped; ordinary users do not need that research snapshot.

This is software equivalence checking, not a new benchmark or biological
validation. Remaining scope differences are enumerated in `method.md`.

The supported release verification environment is listed in `installation.md`.
Earlier checks using PyTorch 2.5.1 were functional comparisons only; they are
superseded for release acceptance by testing on the patched dependency floor.
Passing equivalence tests is not evidence that loading untrusted artifacts is
safe. See `dependency-security.md` for the loading requirements and advisory scope.
