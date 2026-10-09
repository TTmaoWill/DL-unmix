"""Reference features, residual network and validation metrics for DL-unmix."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


EPS = 1e-8


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def transform_expression(values):
    values = np.asarray(values, dtype=np.float32)
    return np.log2(np.clip(values, 0.0, None) + 1.0).astype(np.float32)


def build_gene_meta(truth, genes, cts):
    anchor = np.stack([
        transform_expression(truth.loc[:, [f"{g}_{ct}" for g in genes]].to_numpy()).mean(axis=0)
        for ct in cts
    ], axis=1).astype(np.float32)
    sd = anchor.std(axis=0, keepdims=True)
    sd[sd < EPS] = 1.0
    standardized = (anchor - anchor.mean(axis=0, keepdims=True)) / sd
    top = np.zeros_like(anchor)
    top[np.arange(len(genes)), anchor.argmax(axis=1)] = 1.0
    ranked = np.partition(anchor, -2, axis=1)
    descriptors = np.stack([
        ranked[:, -1] - ranked[:, -2], anchor.std(axis=1), anchor.mean(axis=1)
    ], axis=1).astype(np.float32)
    descriptor_sd = descriptors.std(axis=0, keepdims=True)
    descriptor_sd[descriptor_sd < EPS] = 1.0
    descriptors = (descriptors - descriptors.mean(axis=0, keepdims=True)) / descriptor_sd
    return {
        "genes": np.asarray(genes, dtype=object), "cts": np.asarray(cts, dtype=object),
        "reference_mean": anchor,
        "gene_features": np.concatenate([standardized, top, descriptors], axis=1).astype(np.float32),
        "local_features": np.stack([standardized, top], axis=2).astype(np.float32),
    }


def build_bundle(bulk, fractions, truth, meta, scalers, fit_scalers, device):
    genes, cts = list(meta["genes"]), list(meta["cts"])
    donors = list(bulk.index)
    d, g, c = len(donors), len(genes), len(cts)
    expression = transform_expression(bulk.loc[:, genes].to_numpy())
    frac = fractions.loc[donors, cts].to_numpy(dtype=np.float32)
    anchor = meta["reference_mean"]
    anchor_linear = np.clip(np.exp2(anchor) - 1.0, 0.0, None).astype(np.float32)
    residual = (expression - transform_expression(frac @ anchor_linear.T)).astype(np.float32)
    if fit_scalers:
        bulk_sd = expression.std(axis=0)
        bulk_sd[bulk_sd < EPS] = 1.0
        residual_sd = float(residual.std())
        scalers = {
            "bulk_mu": expression.mean(axis=0).astype(np.float32), "bulk_sd": bulk_sd.astype(np.float32),
            "resid_mu": float(residual.mean()), "resid_sd": residual_sd if residual_sd >= EPS else 1.0,
        }
    bulk_scaled = ((expression - scalers["bulk_mu"]) / scalers["bulk_sd"]).astype(np.float32)
    residual_scaled = ((residual - float(scalers["resid_mu"])) / float(scalers["resid_sd"])).astype(np.float32)
    donor = np.concatenate([
        bulk_scaled.reshape(-1, 1), residual_scaled.reshape(-1, 1), np.repeat(frac, g, axis=0)
    ], axis=1)
    tensors = {
        "donor_feat": donor,
        "gene_feat": np.tile(meta["gene_features"], (d, 1)),
        "local_feat": np.tile(meta["local_features"], (d, 1, 1)),
        "ref_mean": np.tile(anchor, (d, 1)),
    }
    truth_frame = None
    if truth is not None:
        columns = [f"{gene}_{ct}" for gene in genes for ct in cts]
        truth_frame = truth.loc[donors, columns]
        tensors["Y_abs"] = transform_expression(truth_frame.to_numpy()).reshape(d*g, c)
    bundle = {key: torch.from_numpy(value).to(device) for key, value in tensors.items()}
    bundle.update(donors=donors, genes=genes, cts=cts, truth_raw_df=truth_frame)
    return bundle, scalers


class ResidualNet(nn.Module):
    """Shared gene/donor representation with a residual head for each cell type."""
    def __init__(self, n_ct, dropout, fraction_floor=1e-6):
        super().__init__()
        self.fraction_floor = fraction_floor
        self.shared = nn.Sequential(
            nn.Linear(5 + 3*n_ct, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 48)
        )
        self.ct_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(51, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
            ) for _ in range(n_ct)
        ])

    def forward(self, donor, gene, local):
        log_fraction = torch.log(donor[:, 2:].clamp(0.0, 1.0).clamp_min(self.fraction_floor))
        shared = self.shared(torch.cat([donor[:, :2], log_fraction, gene], dim=1))
        return torch.cat([
            head(torch.cat([shared, log_fraction[:, i:i+1], local[:, i, :]], dim=1))
            for i, head in enumerate(self.ct_towers)
        ], dim=1)


def build_row_dataloader(dataset, batch_size, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def build_pair_indices(n_ct):
    return tuple(torch.triu_indices(n_ct, n_ct, offset=1))


def training_loss(delta, truth, anchor, pair_i, pair_j):
    predicted = anchor + delta
    predicted_pairs = predicted.index_select(1, pair_i) - predicted.index_select(1, pair_j)
    truth_pairs = truth.index_select(1, pair_i) - truth.index_select(1, pair_j)
    return F.smooth_l1_loss(delta, truth-anchor, beta=1.0) + F.smooth_l1_loss(predicted_pairs, truth_pairs, beta=1.0)


def predict_to_frame(model, bundle, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(bundle["donor_feat"]), batch_size):
            rows = slice(start, start + batch_size)
            delta = model(bundle["donor_feat"][rows], bundle["gene_feat"][rows], bundle["local_feat"][rows])
            predictions.append((bundle["ref_mean"][rows] + delta).cpu())
    columns = [f"{g}_{ct}" for g in bundle["genes"] for ct in bundle["cts"]]
    values = torch.cat(predictions).numpy().reshape(len(bundle["donors"]), len(columns))
    return pd.DataFrame(values, index=pd.Index(bundle["donors"], name="donor"), columns=columns)


def corrcoef_safe(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x, y = x[mask], y[mask]
    x, y = x-x.mean(), y-y.mean()
    denominator = float(np.sqrt((x*x).sum() * (y*y).sum()))
    return float((x*y).sum() / denominator) if denominator > 0 else float("nan")


def evaluate_val_pcc(model, bundle, batch_size):
    predicted = predict_to_frame(model, bundle, batch_size)
    truth = bundle["truth_raw_df"]
    rows = []
    for ct in bundle["cts"]:
        correlations = np.asarray([
            corrcoef_safe(transform_expression(truth[f"{g}_{ct}"].to_numpy()), predicted[f"{g}_{ct}"].to_numpy())
            for g in bundle["genes"]
        ])
        finite = correlations[np.isfinite(correlations)]
        rows.append({"cell_type": ct, "n_genes": len(finite),
                     "median_pcc": float(np.median(finite)) if len(finite) else float("nan")})
    scores = [row["median_pcc"] for row in rows if np.isfinite(row["median_pcc"])]
    score = float(np.mean(scores)) if scores else float("nan")
    return pd.DataFrame(rows), score, predicted.reset_index()
