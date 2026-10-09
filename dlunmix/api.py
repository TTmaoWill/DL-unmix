"""Reference-supervised fitting and truth-free prediction.

Expression inputs are normalized linear expression, before log2(max(x,0)+1).
Reference CTS is a mapping from cell type to donor-by-gene DataFrame.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from . import _reference as ref


@dataclass(frozen=True)
class FitConfig:
    """Adopted defaults; candidate epochs may be shortened for software demos."""
    candidate_epochs: tuple[int, ...] = (5, 10, 15, 20)
    seed: int = 20260307
    batch_size: int = 4096
    eval_batch_size: int = 8192
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    dropout: float = 0.15

    def __post_init__(self):
        if (not self.candidate_epochs or any(type(x) is not int or x <= 0 for x in self.candidate_epochs)
                or tuple(sorted(set(self.candidate_epochs))) != tuple(self.candidate_epochs)):
            raise ValueError("candidate_epochs must be strictly increasing positive integers")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if any(type(x) is not int or x <= 0 for x in (self.batch_size, self.eval_batch_size)):
            raise ValueError("batch sizes must be positive integers")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not np.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


def _labels(values, name):
    values = list(values)
    if not values or any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError(f"{name} must contain nonempty string labels")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate labels")
    return values


def _frame(frame, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    _labels(frame.index, f"{name} donors")
    _labels(frame.columns, f"{name} columns")
    try:
        values = frame.to_numpy(dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} must contain numeric values") from e
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing, infinite or out-of-range values")
    return pd.DataFrame(values, index=frame.index.copy(), columns=frame.columns.copy())


def _fractions(frame, donors, cell_types=None):
    frame = _frame(frame, "fractions")
    if set(frame.index) != set(donors):
        raise ValueError("fraction donor labels must exactly match bulk donor labels")
    if cell_types is not None and set(frame.columns) != set(cell_types):
        raise ValueError("fraction cell types must exactly match the fitted cell types")
    frame = frame.loc[donors, cell_types if cell_types is not None else frame.columns]
    a = frame.to_numpy()
    if (a < 0).any() or (a > 1).any() or not np.allclose(a.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("fractions must be in [0,1] and rows must sum to one; no implicit renormalization")
    return frame


def _truth(cts, donors, genes, cell_types):
    if not isinstance(cts, Mapping) or set(cts) != set(cell_types):
        raise ValueError("reference/evaluation CTS must map each cell type to a DataFrame")
    blocks = []
    for ct in cell_types:
        f = _frame(cts[ct], f"CTS[{ct}]")
        if set(f.index) != set(donors) or set(f.columns) != set(genes):
            raise ValueError("CTS donor and gene labels must exactly match the corresponding bulk/prediction")
        f = f.loc[donors, genes].copy()
        f.columns = [f"{g}_{ct}" for g in genes]
        blocks.append(f)
    out = pd.concat(blocks, axis=1)
    if not out.columns.is_unique:
        raise ValueError("gene/cell-type labels collide when joined by underscore; use unambiguous labels")
    return out


def _network(n_ct, config, floor=1e-6):
    return ref.EarlySplitResidualNet(
        donor_dim=3 + 2*n_ct, gene_dim=2*n_ct + 3, local_dim=4,
        output_dim=n_ct, dropout=config.dropout, ct_names=[str(i) for i in range(n_ct)],
        bulk_feature_mode="bulk_resid", fraction_feature_mode="log_only",
        fraction_input_scope="both", fraction_min_clip=floor,
    )


def _bundle(bulk, fractions, truth, meta, scalers=None, fit=False, floor=1e-6):
    return ref.build_bundle(bulk, fractions, truth, meta, scalers, fit,
                            torch.device("cpu"), bulk_feature_mode="bulk_resid", fraction_min_clip=floor)


def _loader(bundle, config):
    dataset = TensorDataset(*(bundle[k] for k in ("donor_feat", "gene_feat", "local_feat", "Y_abs", "ref_mean")))
    return ref.build_row_dataloader(dataset, bundle, config.batch_size, config.seed)


def _train_epoch(model, loader, optimizer, pair_i, pair_j):
    model.train()
    losses = []
    for donor, gene, local, truth, anchor in loader:
        optimizer.zero_grad()
        delta, predicted, _ = ref.forward_pred(model, donor, gene, local, anchor)
        loss, _ = ref.compute_losses(delta, predicted, truth, anchor, pair_i, pair_j, 1.0, 1.0, 1.0)
        if not torch.isfinite(loss):
            raise ValueError("non-finite training loss; check expression input scale")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _pcc_array(predicted, truth, genes, cell_types):
    return np.asarray([[ref.corrcoef_safe(
        ref.transform_expression_numpy(truth[f"{g}_{ct}"].to_numpy(), "log2p1_nonnegative"),
        predicted[f"{g}_{ct}"].to_numpy()) for ct in cell_types] for g in genes], dtype=np.float64)


class DLUnmix:
    """CPU implementation of the adopted model with a portable fitted artifact.

    Fit splits must be supplied explicitly. Predict returns unfiltered expression
    on the processed scale with (gene, cell_type) MultiIndex columns.
    """
    def __init__(self, config: FitConfig | None = None):
        self.config = config or FitConfig()

    def fit(self, reference_bulk: pd.DataFrame, reference_fractions: pd.DataFrame,
            reference_cts: Mapping[str, pd.DataFrame], *, train_donors: Sequence[str],
            validation_donors: Sequence[str], refit_only_donors: Sequence[str] = ()):
        """Fit atomically, retaining any existing fitted model if fitting fails."""
        candidate = type(self)(self.config)
        candidate._fit(reference_bulk, reference_fractions, reference_cts,
                       train_donors=train_donors, validation_donors=validation_donors,
                       refit_only_donors=refit_only_donors)
        self.__dict__ = candidate.__dict__.copy()
        return self

    def _fit(self, reference_bulk, reference_fractions, reference_cts, *,
             train_donors, validation_donors, refit_only_donors):
        bulk = _frame(reference_bulk, "reference bulk")
        genes = sorted(bulk.columns)
        bulk = bulk.loc[:, genes]
        fractions = _fractions(reference_fractions, list(bulk.index))
        cell_types = list(fractions.columns)
        if len(cell_types) < 2:
            raise ValueError("at least two cell types are needed for the contrast loss")
        train = _labels(train_donors, "training donors")
        validation = _labels(validation_donors, "validation donors")
        extra = _labels(refit_only_donors, "refit-only donors") if len(refit_only_donors) else []
        all_split = train + validation + extra
        if len(all_split) != len(set(all_split)) or set(all_split) != set(bulk.index):
            raise ValueError("splits must be disjoint and together cover every reference donor")
        if len(train) < 2 or len(validation) < 2:
            raise ValueError("at least two training and two validation donors are required")
        truth = _truth(reference_cts, list(bulk.index), genes, cell_types)
        # Preserve the supplied split-file order, as in archived train_step1.
        cfg = self.config
        ref.set_determinism(cfg.seed)
        meta = ref.build_gene_meta(truth.loc[train], genes, cell_types)
        train_bundle, scalers = _bundle(bulk.loc[train], fractions.loc[train], truth.loc[train], meta, fit=True)
        val_bundle, _ = _bundle(bulk.loc[validation], fractions.loc[validation], truth.loc[validation], meta, scalers)
        model = _network(len(cell_types), cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        loader = _loader(train_bundle, cfg)
        pairs = ref.build_pair_indices(len(cell_types))
        best_score = -np.inf
        best_predictions = None
        history = []
        for epoch in range(1, max(cfg.candidate_epochs) + 1):
            _train_epoch(model, loader, optimizer, *pairs)
            if epoch not in cfg.candidate_epochs:
                continue
            by_ct, score, prediction = ref.evaluate_val_pcc(model, val_bundle, cfg.eval_batch_size)
            history.append({"epoch": epoch, "mean_median_signed_pcc": float(score) if np.isfinite(score) else None,
                            "finite_genes_per_cell_type": [int(x) for x in by_ct.n_genes]})
            if np.isfinite(score) and score > best_score:
                best_score, self.selected_epochs_ = score, epoch
                best_predictions = prediction.set_index(prediction.columns[0])
        if best_predictions is None:
            raise ValueError("all validation scores are undefined; provide variable CTS profiles and enough validation donors")
        self.validation_pcc_ = _pcc_array(best_predictions, truth.loc[validation], genes, cell_types)
        self.history_ = history
        self.reference_counts_ = {"train": len(train), "validation": len(validation), "refit_only": len(extra), "total": len(bulk)}
        # A fresh seeded model, optimizer, anchors and scalers on all reference donors.
        ref.set_determinism(cfg.seed)
        self.gene_meta_ = ref.build_gene_meta(truth, genes, cell_types)
        full, self.scalers_ = _bundle(bulk, fractions, truth, self.gene_meta_, fit=True)
        self.model_ = _network(len(cell_types), cfg)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        loader = _loader(full, cfg)
        for _ in range(self.selected_epochs_):
            _train_epoch(self.model_, loader, optimizer, *pairs)
        self.model_.eval()
        self.genes_, self.cell_types_ = genes, cell_types
        return self

    def _check_fitted(self):
        if not hasattr(self, "genes_"):
            raise ValueError("fit or load a model before prediction or saving")

    def predict(self, bulk: pd.DataFrame, fractions: pd.DataFrame, *, fraction_floor: float = 0.01):
        """Predict processed CTS expression; no target truth or refitting is used.

        The default 0.01 is the adopted accuracy/deployment floor. Use 1e-6
        explicitly for the original-input setting. Original fractions still
        enter the composition residual unchanged. All fitted genes are required.
        """
        self._check_fitted()
        if not np.isfinite(fraction_floor) or not 0 < fraction_floor <= 1:
            raise ValueError("fraction_floor must be finite and in (0,1]")
        bulk = _frame(bulk, "target bulk")
        if set(bulk.columns) != set(self.genes_):
            raise ValueError("target genes must exactly match the fitted panel; subset explicitly before prediction")
        bulk = bulk.loc[:, self.genes_]
        fractions = _fractions(fractions, list(bulk.index), self.cell_types_)
        bundle, _ = _bundle(bulk, fractions, None, self.gene_meta_, self.scalers_, floor=fraction_floor)
        # The reference forward path recomputes log fractions, so set both paths.
        previous_floor = self.model_.fraction_min_clip
        try:
            self.model_.fraction_min_clip = float(fraction_floor)
            pred = ref.predict_to_frame(self.model_, bundle, self.config.eval_batch_size)
        finally:
            self.model_.fraction_min_clip = previous_floor
        pred.columns = pd.MultiIndex.from_product([self.genes_, self.cell_types_], names=["gene", "cell_type"])
        return pred

    def selected_profiles(self, threshold: float = 0.4):
        """Signed reference validation PCC > threshold; does not change predictions."""
        self._check_fitted()
        if not np.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("threshold must be finite and within [-1,1]")
        return pd.DataFrame(self.validation_pcc_ > threshold, index=self.genes_, columns=self.cell_types_)

    def save(self, directory):
        """Write a new model directory; never overwrite an existing artifact."""
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        metadata = {"format_version": 1, "package_version": "0.2.0", "variant": "V0_no_expected_logfrac_only",
                    "config": asdict(self.config), "genes": self.genes_, "cell_types": self.cell_types_,
                    "selected_epochs": self.selected_epochs_, "history": self.history_, "reference_counts": self.reference_counts_,
                    "training_fraction_floor": 1e-6, "default_prediction_fraction_floor": 0.01,
                    "expression_transform": "log2p1_nonnegative"}
        arrays = {"meta__" + k: v for k, v in self.gene_meta_.items() if k not in ("genes", "cts")}
        arrays.update({"scaler__" + k: np.asarray(v) for k, v in self.scalers_.items()})
        arrays["validation_pcc"] = self.validation_pcc_
        np.savez_compressed(directory / "features.npz", **arrays)
        torch.save({k: v.detach().cpu() for k, v in self.model_.state_dict().items()}, directory / "weights.pt")
        (directory / "model.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, directory):
        """Load a locally fitted artifact (JSON, non-pickled arrays, tensor weights)."""
        directory = Path(directory)
        d = json.loads((directory / "model.json").read_text())
        if d.get("format_version") != 1 or d.get("variant") != "V0_no_expected_logfrac_only":
            raise ValueError("unsupported model artifact format or variant")
        if d.get("expression_transform") != "log2p1_nonnegative" or d.get("training_fraction_floor") != 1e-6:
            raise ValueError("unsupported model preprocessing")
        cfg = dict(d["config"])
        cfg["candidate_epochs"] = tuple(cfg["candidate_epochs"])
        obj = cls(FitConfig(**cfg))
        obj.genes_ = _labels(d["genes"], "model genes")
        obj.cell_types_ = _labels(d["cell_types"], "model cell types")
        if len(obj.cell_types_) < 2:
            raise ValueError("model must contain at least two cell types")
        with np.load(directory / "features.npz", allow_pickle=False) as a:
            obj.gene_meta_ = {k[6:]: a[k].copy() for k in a.files if k.startswith("meta__")}
            obj.scalers_ = {k[8:]: a[k].copy() for k in a.files if k.startswith("scaler__")}
            obj.validation_pcc_ = a["validation_pcc"].copy()
        obj.gene_meta_.update(genes=np.asarray(obj.genes_, dtype=object), cts=np.asarray(obj.cell_types_, dtype=object))
        if obj.gene_meta_["ref_mean_raw"].shape != (len(obj.genes_), len(obj.cell_types_)):
            raise ValueError("model anchor shape does not match labels")
        obj.model_ = _network(len(obj.cell_types_), obj.config)
        obj.model_.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True), strict=True)
        obj.model_.eval()
        obj.selected_epochs_, obj.history_, obj.reference_counts_ = d["selected_epochs"], d["history"], d["reference_counts"]
        return obj


def evaluate(predictions: pd.DataFrame, truth: Mapping[str, pd.DataFrame]):
    """Optional per-profile signed PCC, absolute PCC and RMSE on processed scale.

    No donor masks, gene filters, significance tests or multiplicity correction
    are inferred. Caller-supplied predictions determine the evaluation population.
    """
    if not isinstance(predictions.columns, pd.MultiIndex) or predictions.columns.nlevels != 2:
        raise ValueError("predictions need (gene, cell_type) MultiIndex columns")
    genes = list(dict.fromkeys(predictions.columns.get_level_values(0)))
    cell_types = list(dict.fromkeys(predictions.columns.get_level_values(1)))
    expected = pd.MultiIndex.from_product([genes, cell_types], names=["gene", "cell_type"])
    if len(predictions.columns) != len(expected) or set(predictions.columns) != set(expected):
        raise ValueError("predictions must contain a complete, unique gene-by-cell-type panel")
    p = predictions.loc[:, expected].copy()
    p.columns = [f"{g}_{ct}" for g, ct in expected]
    p = _frame(p, "predictions")
    t = _truth(truth, list(p.index), genes, cell_types)
    rows = []
    for g, ct in expected:
        y = ref.transform_expression_numpy(t[f"{g}_{ct}"].to_numpy(), "log2p1_nonnegative")
        pred = p[f"{g}_{ct}"].to_numpy(dtype=np.float64)
        corr = ref.corrcoef_safe(y, pred)
        rows.append({"gene": g, "cell_type": ct, "n_donors": len(p), "pcc": corr,
                     "abs_pcc": abs(corr), "rmse": float(np.sqrt(np.mean((y.astype(float)-pred)**2)))})
    return pd.DataFrame(rows)
