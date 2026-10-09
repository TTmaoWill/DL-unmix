# DL-unmix

Predict donor-level cell-type-specific gene expression from bulk RNA-seq data.

## Introduction

DL-unmix learns from a single-cell or single-nucleus reference summarized by
donor and cell type. It combines bulk expression, cell-type proportions and
reference expression profiles in a neural network to predict cell-type-specific
(CTS) expression for each target donor. Training selects the number of epochs
using held-out reference donors, then fits the final model on the full reference.

## Installation

Install from a source checkout in a Python environment:

```bash
git clone https://github.com/TTmaoWill/DL-unmix.git
cd DL-unmix
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The installer checks the Python requirement and installs NumPy, pandas and
PyTorch. Computation uses CPU by default. To use an available CUDA device, add
`--device cuda` to a command.

## Prepare the data

Training requires:

- Reference bulk expression: a donor-by-gene matrix.
- Reference cell-type proportions: a donor-by-cell-type matrix.
- Reference CTS expression: a donor-by-gene matrix for each cell type.
- A donor split identifying training and validation samples.

Prediction requires target bulk expression and cell-type proportions, with the
same genes and cell types as the fitted model. Prepare expression on matched,
normalized linear scales; DL-unmix applies `log2(max(expression, 0) + 1)`
internally. Supply proportions between zero and one, summing to one per donor.
Use distinct reference and target individuals when assessing prediction accuracy.

See [input formats](docs/input-format.md) for the TSV layouts and Python inputs.
The example below generates small files in each required format.

## Run an example

```bash
dlunmix demo --out demo-output
```

This creates synthetic reference and target data, trains a small model, saves
and reloads it, and predicts target CTS expression. Results are written to
`demo-output/prediction/`; the fitted model is in `demo-output/model/`.
Choose a new output directory for each run.

## Train a model

The following command uses the example inputs. Replace these paths with your
own prepared data to train on a real reference panel.

```bash
dlunmix fit \
  --reference-bulk demo-output/reference_bulk.tsv \
  --reference-fractions demo-output/reference_fractions.tsv \
  --reference-cts demo-output/reference_cts.tsv \
  --splits demo-output/splits.tsv \
  --out fitted-model
```

## Predict CTS expression

```bash
dlunmix predict \
  --model fitted-model \
  --bulk demo-output/target_bulk.tsv \
  --fractions demo-output/target_fractions.tsv \
  --out target-prediction
```

`target-prediction/predictions.tsv` contains expression on the transformed log
scale for every fitted gene and cell type. `selected_profiles.tsv` separately
identifies profiles with signed validation PCC above 0.4. Prediction uses a
fraction floor of 0.01 in log-fraction features; set `--fraction-floor` explicitly
when a different value is required by your analysis.

The same saved model can be used from Python:

```python
from dlunmix import DLUnmix
from dlunmix.cli import read_matrix

model = DLUnmix.load("fitted-model")
bulk = read_matrix("demo-output/target_bulk.tsv")
fractions = read_matrix("demo-output/target_fractions.tsv")
predictions = model.predict(bulk, fractions)
```

For larger cohorts, predict in donor batches sized to fit available memory.
See the [method description](docs/method.md) for the architecture and training
procedure, and [development notes](CONTRIBUTING.md) for tests and model conversion.

## License

DL-unmix is distributed under the [MIT license](LICENSE).
