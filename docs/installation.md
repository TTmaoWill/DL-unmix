# Installation and development

Package installation reads only `pyproject.toml`. The existing R requirements,
benchmark scripts and `src/` directory are legacy and are not package dependencies.
Python >=3.10 is declared; the release verification environment uses Python
3.11.14, NumPy 2.3.5, pandas 2.3.3 and PyTorch 2.10.0+cpu. Other compatible
versions are not claimed to have been tested.

PyTorch >=2.10.0,<3 is required. This floor includes fixes for two known
`weights_only=True` loading vulnerabilities; see
[dependency security](dependency-security.md). Earlier functional checks on
PyTorch 2.5.1 do not establish release security and that version is unsupported.

For a Linux CPU environment matching the verification configuration, install
the official CPU wheel before the package:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

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
checkpoints. Load only artifacts from trusted sources: neither the version floor
nor restricted deserialization makes arbitrary untrusted checkpoints safe.
Concurrent mutation or
prediction with different fraction floors or devices on one model instance is unsupported;
use separate loaded instances per concurrent worker.

## Device verification coverage

The device extension is checked on the CPU environment above, including default
versus explicit CPU training, original-script parity, artifact roundtrips,
CLI device selection and explicit rejection of unavailable CUDA. The test suite
includes two conditional CUDA checks: fixed-weight CPU/CUDA inference plus
portable loading, and tiny CUDA training with tensor/index placement assertions.
These CUDA checks are skipped when CUDA is unavailable. No GPU execution or
CPU/CUDA numerical equivalence has yet been verified for this extension.

To run those conditional checks, use a CUDA-enabled PyTorch >=2.10.0,<3 build
compatible with the allocated GPU and driver, then run the regular test suite
inside that allocation. Cluster login nodes are not compute resources. CPU
training and CUDA training can differ because of random-number streams and
floating-point kernels, even with the same seed. GPU inference comparisons use
rtol=1e-5 and atol=2e-5, not bitwise equality. No speedup claim is made.
