# Loading fitted models

Version 0.3.0 loads format-2 model directories containing three files:

| File | Contents |
| --- | --- |
| `model.json` | Gene and cell-type labels, fitting configuration, selected epochs and aggregate reference counts |
| `features.npz` | Reference features, normalization parameters and per-profile validation PCC |
| `weights.pt` | CPU tensor weights, portable across supported CPU/CUDA environments |

Keep all three files together. Use trusted artifacts with the supported
PyTorch version described in [installation](installation.md).

```python
from dlunmix import DLUnmix

model = DLUnmix.load("fitted-model", device="cpu")
# bulk and fractions are donor-by-feature pandas DataFrames.
bulk = bulk.loc[:, model.genes_]
fractions = fractions.loc[bulk.index, model.cell_types_]
prediction = model.predict(bulk, fractions, fraction_floor=0.01)
selected = model.selected_profiles(threshold=0.4)
```

Supply every fitted gene and cell type. Normalize expression to compatible
linear units before calling the model; do not pass logged expression.
Fractions must lie in [0,1] and sum to one per donor. Prediction needs neither
the original reference samples nor target cell-type expression labels.
The separate selection mask does not zero or remove predictions.

The CLI exposes the same floor explicitly:

```bash
dlunmix predict --model fitted-model \
  --bulk target_bulk.tsv --fractions target_fractions.tsv \
  --fraction-floor 0.01 --device cpu --out prediction-output
```

For large cohorts, predict successive donor batches with the same fitted model
and concatenate them in the original donor order. The model does not refit
normalization parameters during prediction.

For a trusted format-1 artifact, run the converter once:

```bash
python tools/convert_model.py old-model fitted-model
```

See [migration](migration.md) for the compatibility boundary. Conversion
preserves learned weights and reference metadata; it does not train a new model.

## Pretrained model availability

No pretrained artifacts are downloadable from this package candidate. The
ROSMAP real-data artifact remains private pending confirmation of its
redistribution permissions. Its historical prediction configuration uses
`fraction_floor=1e-6`; substituting the API's 0.01 default changes that setting.
The [validation record](validation.md) describes the completed numerical check.
