# Development

Install a source checkout in an environment satisfying `pyproject.toml`:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Tests cover input validation, deterministic fitting, prediction, model
serialization, fraction floors, CLI workflows and device placement. CUDA tests
are skipped when CUDA is unavailable.

## Reference parity

The optional numerical parity tests compare features, prediction, training and
model conversion with an independently archived implementation. Set
`DLUNMIX_REFERENCE_SOURCE` to a directory containing `dl_unmix_common.py` to run
these tests. Its SHA-256 must match `tests/reference_source.json`; without this
source, the two tests are skipped. The archived implementation is not bundled.

```bash
DLUNMIX_REFERENCE_SOURCE=/path/to/reference/source \
  python -m unittest discover -s tests -v
```

## Build distributions

```bash
python -m pip install build
python -m build
```

`pyproject.toml` is the canonical source of dependency requirements and package
metadata. The source distribution includes tests, documentation and utilities;
the wheel contains the Python package and license files.

## Convert an existing model

The package reads format-2 models. Convert a trusted format-1 model once using
the utility in this checkout:

```bash
python tools/convert_model.py old-model converted-model
```

Conversion transfers fitted weights, reference features, scalers, validation
correlations and training metadata, then validates and saves a new directory.
It does not retrain the model. Keep the source until predictions have been
checked on your inputs. Floating-point comparisons use `rtol=1e-5` and
`atol=2e-5`; changing matrix dimensions can change accumulation order.

See [dependency security](docs/dependency-security.md) for model-loading guidance.
