# Source provenance and verification

The adopted implementation is the `DL_source` snapshot accompanying the
`fujita_mathys_no_overlap_all_methods_20261005` rerun, referenced by the adopted
`figure2_disjoint_reference_20261008` source manifest. The latter records audit
and result hashes, not a historical hash of the Python source itself. This
release records the inspected source hashes in `provenance.json`; it does not
assert an independently established historical byte identity.

`dlunmix/_model.py` implements reference features, the shared representation,
cell-type heads, the two training losses and validation scoring. The original
source hashes identify the numerical reference used to check this implementation.

## Software verification

The regular unittest suite covers fit/save/load/predict, deterministic seeded
training, label alignment, invalid inputs, fraction floors, output selection,
optional evaluation and CLI file roundtrips. It generates all data at runtime.

To run the additional reference parity check, provide a local directory
containing `dl_unmix_common.py` matching the recorded hash:

```bash
DLUNMIX_REFERENCE_SOURCE=/path/to/adopted/DL_source \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m unittest discover -s release_tests -v
```

The test compares reference features and scalers exactly, then checks prediction
from converted weights for 2, 3 and 5 cell types, including zero fractions and
both supported fraction floors. Three training epochs use aligned initial weights,
row order and dropout random states; losses, parameters, predictions and validation
scores are compared. It also checks artifact conversion and reload. Numerical
comparisons use rtol=1e-5 and atol=2e-5. The test reads the supplied source and uses
generated synthetic data. Without the environment variable it is explicitly skipped.

Fresh seeded training is reproducible within this implementation. Input dimensions
set the Linear initializer's scale and random-number consumption, so retraining
with a seed from a different architecture does not recreate its fitted model.
Use converted trained weights when preserving an existing model's predictions.

The supported release verification environment is listed in `installation.md`.
Earlier checks using PyTorch 2.5.1 were functional comparisons only; they are
superseded for release acceptance by testing on the patched dependency floor.
Passing equivalence tests is not evidence that loading untrusted artifacts is
safe. See `dependency-security.md` for the loading requirements and advisory scope.
