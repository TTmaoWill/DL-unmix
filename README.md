# DL-unmix

DL-unmix learns donor-level cell-type-specific (CTS) expression from a
donor-resolved reference, then predicts CTS expression from target bulk expression
and cell-type fractions. This release exposes the current
`V0_no_expected_logfrac_only` method through a Python API and command line.
It does not estimate cell fractions. Target CTS ground truth is optional and is
used only for evaluation.

## Install

Use Python 3.10 or newer in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
dlunmix --help
```

Core dependencies are NumPy, pandas and PyTorch >=2.10.0,<3. R, comparator methods, GPU
support and plotting libraries are not required. The public interface currently
runs on CPU. `requirements.txt` delegates to the same package metadata.
See [installation](docs/installation.md) for the tested environment and build checks,
and [dependency security](docs/dependency-security.md) for the version floor and
trusted-model requirements.

## Run the synthetic example

```bash
dlunmix demo --out demo-output
```

This generates tiny synthetic reference/target inputs, fits and reloads a model,
and writes predictions and optional evaluation. It uses candidate epochs 1 and 2
for a short software demonstration, not the method's default 5, 10, 15 and 20.
It is not evidence of biological accuracy. Outputs go to a new directory;
existing artifacts are never overwritten.

## Fit and predict your own data

Input expression must already be normalized on the **linear scale**, before
`log2(max(x,0)+1)`. Do not supply already logged expression. Bulk and CTS
reference matrices must use compatible units. Prepare a fixed common gene panel
before fitting; do not choose it using target CTS labels.

```bash
dlunmix fit \
  --reference-bulk reference_bulk.tsv \
  --reference-fractions reference_fractions.tsv \
  --reference-cts reference_cts.tsv \
  --splits splits.tsv --out fitted-model

dlunmix predict --model fitted-model \
  --bulk target_bulk.tsv --fractions target_fractions.tsv \
  --out predicted
```

The demo produces files in exactly these formats. Bulk/fraction TSV files have
donors in rows and named genes/cell types in columns. CTS TSV files have two
header rows (gene, cell type), with donor IDs in the first column. `splits.tsv`
has `donor` and `split` columns (`train`, `val`, optionally `refit_only`). See
[input formats](docs/input-format.md), including normalization and missing cells.

Prediction writes all profiles on the processed expression scale. A separate
`selected_profiles.tsv` identifies signed reference-validation PCC > 0.4;
unselected profiles are not replaced by zero. Pass `--truth-cts` only if you
want an additional per-profile evaluation table.

Python usage:

```python
from dlunmix import DLUnmix

# bulk: donor x gene DataFrame; fractions: donor x cell-type DataFrame
# reference_cts: {cell_type: donor x gene DataFrame}
model = DLUnmix().fit(
    reference_bulk, reference_fractions, reference_cts,
    train_donors=train_ids, validation_donors=validation_ids,
)
model.save("fitted-model")
restored = DLUnmix.load("fitted-model")
predicted = restored.predict(target_bulk, target_fractions)
selected = restored.selected_profiles(threshold=0.4)
```

## Method and scope

The model adds predicted donor-specific residuals to reference CTS means.
Training uses a shared 64/48 network, cell-type heads 64/32/1, dropout 0.15,
two equally weighted Smooth L1 losses, and AdamW. Training duration is selected
by the mean across cell types of median signed gene-wise validation PCC. A
fresh model is then fitted to all reference donors for the chosen duration.
Reference anchors and scalers are recomputed for that full-reference fit.

Training and reference validation use a log-fraction floor of **1e-6**.
Prediction defaults to **0.01**, matching the adopted accuracy/deployment setting;
pass `--fraction-floor 1e-6` explicitly for original-input predictions.
Original fractions remain unchanged in the composition-residual calculation.
See [method details and release differences](docs/method.md).

Performance depends on reference coverage, input normalization and fraction
accuracy. Predicted expression alone does not establish valid CTS disease
effects or controlled downstream false discovery rates. This package does not
apply paper-specific target donor screens, run DEG/eQTL analyses, or reproduce
benchmark significance tests. No participant data or pretrained research models
are distributed. Locally fitted artifacts contain reference-derived information
and should be handled under the source data's access conditions.

## Tests and provenance

```bash
python -m unittest discover -s release_tests -v
```

An optional parity test can compare the release with the adopted original
implementation using only generated synthetic data; see
[provenance](docs/provenance.md). Source-file hashes are recorded in
[provenance.json](docs/provenance.json).

The older `src/`, `script/`, R/comparator scripts and related files remain in
Git history and in the checkout as **legacy research code**. They are not part
of the installed package and are not imported by `dlunmix`.

## License

The current release files are provided under the [MIT license](LICENSE).
[LICENSE_SCOPE.md](LICENSE_SCOPE.md) identifies the covered files and separates
the retained multi-author legacy code; this release does not relicense that code
or its dependencies. Publication citation metadata will be added when an
author-approved paper citation is available.
