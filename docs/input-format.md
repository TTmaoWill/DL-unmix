# Input and output formats

## Expression and fractions

The Python API accepts pandas DataFrames with nonempty unique string labels.
Bulk matrices are donors × genes; fraction matrices are donors × cell types.
Reference CTS is a mapping `{cell_type: DataFrame}`, with the same reference
donors and genes in each matrix. All values must be numeric and finite.
No missing donor, gene or cell type is silently dropped or imputed.
Label order may differ: the interface explicitly aligns labels. Gene order is
sorted when fitting; cell-type order follows the reference fraction columns.

Expression is normalized **before** calling DL-unmix. Human pseudobulk inputs
used full-genome per-cell CPM normalization before selecting genes, then
arithmetic means within donor/cell type and within donor for bulk. This API
starts from those aggregated linear-scale matrices. It does not normalize raw
counts, convert between assays or reconstruct single-cell references. Other
already normalized linear scales require appropriate reference/target matching.

The adopted transformation is `log2(max(expression, 0) + 1)`. Negative inputs
are clipped by that transformation; they are not interpreted as already logged
values. Predictions are on this processed scale and are not forcibly clipped.
The inverse transform `max(2**prediction - 1, 0)` is used internally for mixing;
returning processed predictions preserves the adopted evaluation scale.

Fractions must be in [0,1], with rows summing to one (absolute tolerance 1e-5).
Zero fractions are allowed. They are not removed and do not create loss masks.
Encoded zero CTS labels for donor/cell-type combinations with no observed cells
remain numerical training targets, as in the adopted implementation. Confirm
this encoding deliberately when constructing your reference.

The fitted gene panel is fixed. Target bulk must contain exactly those genes;
explicitly subset extra genes before prediction. Missing genes cause an error.
Targets must have the same cell types and donor labels must match fractions.
DL-unmix requires at least two cell types because the objective includes
cell-type contrasts. It does not choose a reference/target intersection using
unavailable target CTS truth.

## CLI TSV files

Bulk and fractions have one header row:

```text
donor	gene001	gene002
personA	1.2	3.4
personB	2.3	4.5
```

Reference CTS and optional target truth use two header rows. The top-left cells
are the header identifiers, followed by paired gene and cell-type labels:

```text
gene	gene001	gene001	gene002	gene002
cell_type	typeA	typeB	typeA	typeB
personA	1.0	2.0	3.0	4.0
personB	2.0	3.0	4.0	5.0
```

Write this format with `dlunmix.cli.write_cts(mapping, path)` or inspect the demo
files. Do not add a third donor-name header row. Numeric-looking donor IDs,
including leading zeros, are preserved. The internal `gene_celltype`
encoding must be unambiguous; colliding label combinations are rejected.

Splits:

```text
donor	split
personA	train
personB	val
```

Supply at least two donors per training/validation group. Splits must be
disjoint and cover all reference donors. `refit_only` explicitly identifies
additional donors used only in the final full-reference fit, and are excluded from held-out validation. The split
file's donor order is retained within training/validation groups. Ensure that
reference and target identities are distinct using the appropriate identity
crosswalk; the software cannot infer that different IDs represent one person.

## Fitted artifacts and prediction

`model.json` records settings, gene/cell-type order, chosen epoch, reference
counts and validation history. `features.npz` contains fitted anchors/scalers
and validation PCCs; `weights.pt` contains CPU tensors. Raw donor matrices and
donor IDs are not stored in the model directory. Loading uses non-pickled NumPy
arrays and PyTorch's `weights_only=True` under the required PyTorch >=2.10.0,<3.
Use artifacts from trusted sources only; these restrictions are not a sandbox
for arbitrary untrusted checkpoints. See [dependency security](dependency-security.md).

`predictions.tsv` has the two-header CTS layout and contains all fitted
profiles. `selected_profiles.tsv` is a separate gene × cell-type Boolean mask.
`prediction.json` records the fraction floor and output scale. Optional
`evaluation.tsv` reports signed PCC, absolute PCC and RMSE by gene/cell type.
Undefined constant-profile PCC is NaN, not zero. Evaluation uses the supplied
donors/profiles without adding paper-specific masks or significance tests.
