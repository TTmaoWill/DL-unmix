# Installation and development

Package installation reads only `pyproject.toml`. The existing R requirements,
benchmark scripts and `src/` directory are legacy and are not package dependencies.
Python >=3.10 is declared; the release verification environment uses Python
3.11.14, NumPy 2.3.5, pandas 2.3.3 and PyTorch 2.5.1 (CPU). Other compatible
versions are not claimed to have been tested.

```bash
python -m pip install .
python -m unittest discover -s release_tests -v
dlunmix demo --out demo-output
```

The wheel contains only `dlunmix` and package metadata. The source distribution
also includes docs, a synthetic example and release tests. No legacy code or
research outputs are included in the distribution artifacts.

For development, use a dedicated environment and `python -m pip install -e .`.
To build distributions with the standard frontend:

```bash
python -m pip install build
python -m build
```

For a provisioned environment that already contains setuptools and wheel,
`python -m pip wheel --no-deps --no-build-isolation .` also builds the wheel.
Use the normal dependency-resolving installation on a new machine. Installing
with `--no-deps` is appropriate only when compatible dependencies are already
present. The short synthetic example requires no external dataset downloads.

Saved models contain a JSON configuration, non-pickled NumPy arrays and tensor
weights. They are versioned artifacts of this interface, not arbitrary legacy
checkpoints. Load only artifacts from trusted sources. Concurrent mutation or
prediction with different fraction floors on one model instance is unsupported;
use separate loaded instances per concurrent worker.
