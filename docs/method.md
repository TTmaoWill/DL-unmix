# DL-unmix model

DL-unmix predicts donor-level cell-type-specific expression by adding learned
residuals to reference expression anchors.

## Features and model

The anchor for each gene/cell type is the arithmetic mean of processed CTS
expression over fitting reference donors. For input features only, anchors are
standardized across genes separately within each cell type (population SD).
SDs below 1e-8 are replaced with 1. The top-type indicator, top-minus-second
gap, across-type SD and across-type mean use the original processed anchors.
The three continuous summaries are subsequently standardized across genes,
separately, with the same safeguard. Descriptor order is standardized anchor
vector, top-type one-hot vector, standardized gap, SD, mean.

Active donor/gene inputs are gene-standardized bulk expression, the globally
standardized composition residual, log fractions and reference descriptors.
The composition residual is processed bulk minus the processed fraction-weighted
mixture of inverse-transformed reference anchors. Each head additionally receives
its log fraction, standardized anchor and top-type indicator. For C cell types, the shared input contains 5 + 3C features.

The shared MLP is Linear→64→ReLU→Dropout(0.15)→Linear→48. The 48-unit shared
output is linear. Each cell-type head appends its three local features and uses hidden layers 64 and 32 with ReLU/dropout, followed by a linear
scalar output. The residual output is added to the original **unstandardized
processed-expression** anchor.

## Training

Each row is a donor/gene combination. Smooth L1 (beta=1) is averaged over residual
prediction errors and separately over all unordered cell-type pair contrasts;
both components have weight 1. No observed-cell-count masks or profile-selection
weights enter training. AdamW uses learning rate 0.0003, weight decay 0.0001,
batch size 4096, evaluation batch size 8192 and seed 20260307.

One reference-training trajectory evaluates epochs 5, 10, 15 and 20. The score
is the mean across cell types of the median finite signed gene-wise PCC on
validation donors. Undefined cell-type medians are excluded by the original
nanmean behavior; finite-gene counts are recorded. All-undefined selection fails
explicitly. Equal best scores retain the earlier epoch. Profile-level validation
PCCs come from the selected training-split model.

The final model is independently initialized after resetting the same seed, then
trained on all reference donors for the selected number of epochs. Anchors and
scalers are recomputed from that full reference. Validation scores are retained
from the held-out procedure and are not recomputed on full-reference fitted
predictions. Signed validation PCC >0.4 defines a separate output-selection mask.

Reference training and validation use a 1e-6 log-fraction floor. Prediction
uses 0.01 by default, configurable through the API or CLI. Floors affect direct
log-fraction features only; fractions retain their original values in the
composition residual.

## Execution

CPU is the default device; CUDA can be selected explicitly. Saved weights use
CPU tensors and can be loaded on either device. Floating-point results can
vary between devices. Inference materializes donor/gene feature arrays, so
memory scales with donors × genes × cell types. Large cohorts can be processed
in donor batches using the same fitted model.
