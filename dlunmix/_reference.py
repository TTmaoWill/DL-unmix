#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler


EPS = 1e-8
DEFAULT_SEED = 20260307
EXPRESSION_TRANSFORM_LOG2P1_NONNEG = "log2p1_nonnegative"
EXPRESSION_TRANSFORM_SIGNED_LOG2P1 = "signed_log2p1"
SUPPORTED_EXPRESSION_TRANSFORMS = (
    EXPRESSION_TRANSFORM_LOG2P1_NONNEG,
    EXPRESSION_TRANSFORM_SIGNED_LOG2P1,
)
FRACTION_TRANSFORM_LEGACY_LOG = "legacy_log"
FRACTION_TRANSFORM_LOG1P_TAU = "log1p_tau"
FRACTION_TRANSFORM_ASIN_SQRT = "asin_sqrt"
FRACTION_TRANSFORM_CLR = "clr"
SUPPORTED_FRACTION_TRANSFORMS = (
    FRACTION_TRANSFORM_LEGACY_LOG,
    FRACTION_TRANSFORM_LOG1P_TAU,
    FRACTION_TRANSFORM_ASIN_SQRT,
    FRACTION_TRANSFORM_CLR,
)


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] {message}", flush=True)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def validate_expression_transform(mode: str) -> str:
    mode = str(mode)
    if mode not in SUPPORTED_EXPRESSION_TRANSFORMS:
        raise ValueError(
            f"Unsupported expression_transform={mode!r}. "
            f"Expected one of {SUPPORTED_EXPRESSION_TRANSFORMS}."
        )
    return mode


def transform_expression_numpy(values: np.ndarray, mode: str) -> np.ndarray:
    mode = validate_expression_transform(mode)
    arr = np.asarray(values, dtype=np.float32)
    if mode == EXPRESSION_TRANSFORM_SIGNED_LOG2P1:
        return (np.sign(arr) * np.log2(np.abs(arr) + 1.0)).astype(np.float32)
    return np.log2(np.clip(arr, a_min=0.0, a_max=None) + 1.0).astype(np.float32)


def inverse_transform_expression_numpy(values: np.ndarray, mode: str) -> np.ndarray:
    mode = validate_expression_transform(mode)
    arr = np.asarray(values, dtype=np.float32)
    if mode == EXPRESSION_TRANSFORM_SIGNED_LOG2P1:
        return (np.sign(arr) * (np.exp2(np.abs(arr)) - 1.0)).astype(np.float32)
    return np.clip(np.exp2(arr) - 1.0, a_min=0.0, a_max=None).astype(np.float32)


def transform_expression_torch(values: torch.Tensor, mode: str) -> torch.Tensor:
    mode = validate_expression_transform(mode)
    if mode == EXPRESSION_TRANSFORM_SIGNED_LOG2P1:
        return torch.sign(values) * torch.log2(torch.abs(values) + 1.0)
    return torch.log2(torch.clamp(values, min=0.0) + 1.0)


def inverse_transform_expression_torch(values: torch.Tensor, mode: str) -> torch.Tensor:
    mode = validate_expression_transform(mode)
    if mode == EXPRESSION_TRANSFORM_SIGNED_LOG2P1:
        return torch.sign(values) * (torch.pow(2.0, torch.abs(values)) - 1.0)
    return torch.clamp(torch.pow(2.0, values) - 1.0, min=0.0)


def validate_fraction_transform(mode: str) -> str:
    mode = str(mode)
    if mode not in SUPPORTED_FRACTION_TRANSFORMS:
        raise ValueError(
            f"Unsupported fraction_transform={mode!r}. "
            f"Expected one of {SUPPORTED_FRACTION_TRANSFORMS}."
        )
    return mode


def transform_fraction_numpy(
    values: np.ndarray,
    mode: str = FRACTION_TRANSFORM_LEGACY_LOG,
    tau: float = 0.01,
    min_clip: float = 1e-6,
) -> np.ndarray:
    mode = validate_fraction_transform(mode)
    arr = np.clip(np.asarray(values, dtype=np.float32), a_min=0.0, a_max=1.0)
    if mode == FRACTION_TRANSFORM_LEGACY_LOG:
        return np.log(np.clip(arr, float(min_clip), None)).astype(np.float32)
    if mode == FRACTION_TRANSFORM_LOG1P_TAU:
        if tau <= 0:
            raise ValueError("fraction_transform_tau must be positive for log1p_tau.")
        return np.log1p(arr / float(tau)).astype(np.float32)
    if mode == FRACTION_TRANSFORM_CLR:
        log_arr = np.log(np.clip(arr, float(min_clip), None))
        return (log_arr - np.mean(log_arr, axis=-1, keepdims=True)).astype(np.float32)
    return np.arcsin(np.sqrt(arr)).astype(np.float32)


def transform_fraction_torch(
    values: torch.Tensor,
    mode: str = FRACTION_TRANSFORM_LEGACY_LOG,
    tau: float = 0.01,
    min_clip: float = 1e-6,
) -> torch.Tensor:
    mode = validate_fraction_transform(mode)
    arr = torch.clamp(values, min=0.0, max=1.0)
    if mode == FRACTION_TRANSFORM_LEGACY_LOG:
        return torch.log(torch.clamp(arr, min=float(min_clip)))
    if mode == FRACTION_TRANSFORM_LOG1P_TAU:
        if tau <= 0:
            raise ValueError("fraction_transform_tau must be positive for log1p_tau.")
        return torch.log1p(arr / float(tau))
    if mode == FRACTION_TRANSFORM_CLR:
        log_arr = torch.log(torch.clamp(arr, min=float(min_clip)))
        return log_arr - torch.mean(log_arr, dim=-1, keepdim=True)
    return torch.asin(torch.sqrt(arr))


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def orient_by_donors(df: pd.DataFrame, donor_ids: list[str]) -> pd.DataFrame:
    donor_set = set(map(str, donor_ids))
    col_hits = len(donor_set.intersection(df.columns.astype(str)))
    row_hits = len(donor_set.intersection(df.index.astype(str)))
    if col_hits > row_hits:
        out = df.T.copy()
        out.index = out.index.astype(str)
        out.columns = out.columns.astype(str)
        return out
    return df


def load_target_tables(bulk_path: Path, truth_cts_path: Path, frac_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frac = load_table(frac_path)
    bulk = orient_by_donors(load_table(bulk_path), frac.index.astype(str).tolist())
    truth = orient_by_donors(load_table(truth_cts_path), frac.index.astype(str).tolist())
    donors = bulk.index.intersection(frac.index).intersection(truth.index)
    return bulk.loc[donors].copy(), truth.loc[donors].copy(), frac.loc[donors].copy()


def load_reference_stage(work_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage = work_dir / "dlunmix_stage"
    frac = load_table(stage / "reference_meanpb_frac.tsv")
    pbs = orient_by_donors(load_table(stage / "reference_meanpb_pbs.tsv"), frac.index.astype(str).tolist())
    cts = orient_by_donors(load_table(stage / "reference_meanpb_cts.tsv"), frac.index.astype(str).tolist())
    donors = pbs.index.intersection(frac.index).intersection(cts.index)
    return pbs.loc[donors].copy(), cts.loc[donors].copy(), frac.loc[donors].copy()


def parse_gene_ct_columns(columns: list[str], cts: list[str]) -> set[str]:
    out = set()
    suffixes = tuple(f"_{ct}" for ct in cts)
    for col in columns:
        if not col.endswith(suffixes):
            continue
        gene, _ = col.rsplit("_", 1)
        out.add(gene)
    return out


def complete_gene_set(cts_df: pd.DataFrame, cts: list[str]) -> set[str]:
    genes = parse_gene_ct_columns(cts_df.columns.astype(str).tolist(), cts)
    return {g for g in genes if all(f"{g}_{ct}" in cts_df.columns for ct in cts)}


def intersect_problem(
    ref_pbs: pd.DataFrame,
    ref_cts: pd.DataFrame,
    ref_frac: pd.DataFrame,
    target_pbs: pd.DataFrame,
    target_cts: pd.DataFrame,
    target_frac: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    cts = [ct for ct in ref_frac.columns.astype(str).tolist() if ct in set(target_frac.columns.astype(str).tolist())]
    genes = (
        set(ref_pbs.columns.astype(str))
        .intersection(target_pbs.columns.astype(str))
        .intersection(complete_gene_set(ref_cts, cts))
        .intersection(complete_gene_set(target_cts, cts))
    )
    return sorted(genes), cts


def make_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    dims = [input_dim] + hidden_dims
    for a, b in zip(dims, dims[1:]):
        layers.append(nn.Linear(a, b))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(dims[-1], output_dim))
    return nn.Sequential(*layers)


def repeated_hidden_dims(hidden_dim: int, num_layers: int) -> list[int]:
    return [int(hidden_dim)] * max(int(num_layers), 1)


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output * ctx.scale, None


def grad_scale(values: torch.Tensor, scale: float) -> torch.Tensor:
    scale = float(scale)
    if scale == 1.0:
        return values
    return _GradScale.apply(values, scale)


class SubjectBalancedBatchSampler(Sampler[list[int]]):
    """Sample equal row counts per donor in each minibatch."""

    def __init__(
        self,
        row_donor_idx: torch.Tensor,
        batch_size: int,
        subjects_per_batch: int = 16,
        rows_per_subject: int = 0,
        seed: int = DEFAULT_SEED,
    ):
        donor_np = row_donor_idx.detach().cpu().numpy().astype(np.int64)
        self.donor_to_rows: dict[int, np.ndarray] = {
            int(donor): np.flatnonzero(donor_np == donor).astype(np.int64) for donor in np.unique(donor_np)
        }
        self.donors = np.asarray(sorted(self.donor_to_rows), dtype=np.int64)
        self.subjects_per_batch = max(1, int(subjects_per_batch))
        if int(rows_per_subject) > 0:
            self.rows_per_subject = int(rows_per_subject)
        else:
            self.rows_per_subject = max(1, int(batch_size) // self.subjects_per_batch)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        donors = self.donors.copy()
        rng.shuffle(donors)
        for start in range(0, len(donors), self.subjects_per_batch):
            batch_donors = donors[start : start + self.subjects_per_batch]
            rows: list[int] = []
            for donor in batch_donors:
                available = self.donor_to_rows[int(donor)]
                replace = self.rows_per_subject > len(available)
                sampled = rng.choice(available, size=self.rows_per_subject, replace=replace)
                rows.extend(int(x) for x in sampled)
            rng.shuffle(rows)
            yield rows

    def __len__(self) -> int:
        return int(np.ceil(len(self.donors) / self.subjects_per_batch))


def build_row_dataloader(
    dataset: torch.utils.data.Dataset,
    bundle: dict[str, object],
    batch_size: int,
    seed: int,
    subject_balanced_batch: bool = False,
    subjects_per_batch: int = 16,
    rows_per_subject: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    if not subject_balanced_batch:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=gen)
    sampler = SubjectBalancedBatchSampler(
        bundle["row_donor_idx"],
        batch_size=batch_size,
        subjects_per_batch=subjects_per_batch,
        rows_per_subject=rows_per_subject,
        seed=seed,
    )
    return DataLoader(dataset, batch_sampler=sampler)


class EarlySplitResidualNet(nn.Module):
    def __init__(
        self,
        donor_dim: int,
        gene_dim: int,
        local_dim: int,
        output_dim: int,
        dropout: float,
        donor_summary_dim: int = 0,
        use_fraction_path_regularization: bool = False,
        fraction_branch_hidden_dim: int = 16,
        fraction_branch_dropout: float = 0.0,
        fraction_branch_scale: float = 1.0,
        use_fraction_confidence: bool = False,
        fraction_confidence_hidden_dim: int = 16,
        use_ct_specific_fraction_confidence: bool = False,
        bulk_feature_mode: str = "full",
        fraction_feature_mode: str = "raw_log",
        fraction_input_scope: str = "both",
        fraction_transform_mode: str = FRACTION_TRANSFORM_LEGACY_LOG,
        fraction_transform_tau: float = 0.01,
        fraction_min_clip: float = 1e-6,
        use_fraction_refinement: bool = False,
        fraction_refinement_hidden_dim: int = 16,
        use_fraction_refiner: bool = False,
        fraction_refiner_hidden_dim: int = 16,
        fraction_refiner_num_layers: int = 1,
        fraction_refiner_dropout: float = 0.0,
        fraction_refiner_eps: float = 1e-6,
        ct_names: list[str] | tuple[str, ...] | None = None,
        donor_fraction_zero_cts: str = "",
        local_fraction_zero_cts: str = "",
        shared_routing_mode: str = "shared_all",
        fraction_gradient_scale: float = 1.0,
        fraction_gate_mode: str = "none",
        fraction_gate_floor: float = 0.0,
        fraction_gate_tau: float = 0.05,
        tower_mode: str = "mlp",
        tower_rank: int = 16,
        tower_deviation_scale: float = 1.0,
    ):
        super().__init__()
        self.n_ct = int(output_dim)
        self.bulk_dim = 3
        self.local_dim = int(local_dim)
        self.donor_summary_dim = int(donor_summary_dim)
        self.use_fraction_path_regularization = bool(use_fraction_path_regularization)
        self.fraction_branch_hidden_dim = int(fraction_branch_hidden_dim)
        self.fraction_branch_dropout = float(fraction_branch_dropout)
        self.fraction_branch_scale = float(fraction_branch_scale)
        self.use_fraction_confidence = bool(use_fraction_confidence)
        self.use_ct_specific_fraction_confidence = bool(use_ct_specific_fraction_confidence)
        self.bulk_feature_mode = str(bulk_feature_mode)
        self.fraction_feature_mode = str(fraction_feature_mode)
        self.fraction_input_scope = str(fraction_input_scope)
        self.fraction_transform_mode = validate_fraction_transform(fraction_transform_mode)
        self.fraction_transform_tau = float(fraction_transform_tau)
        self.fraction_min_clip = float(fraction_min_clip)
        self.use_fraction_refinement = bool(use_fraction_refinement)
        self.use_fraction_refiner = bool(use_fraction_refiner)
        self.fraction_refiner_eps = float(fraction_refiner_eps)
        self.ct_names = [str(x) for x in (ct_names or [])]
        self.shared_routing_mode = str(shared_routing_mode)
        self.fraction_gradient_scale = float(fraction_gradient_scale)
        self.fraction_gate_mode = str(fraction_gate_mode)
        self.fraction_gate_floor = float(fraction_gate_floor)
        self.fraction_gate_tau = float(fraction_gate_tau)
        self.tower_mode = str(tower_mode)
        self.tower_rank = int(tower_rank)
        self.tower_deviation_scale = float(tower_deviation_scale)
        if self.bulk_feature_mode not in {
            "full",
            "bulk_resid",
            "bulk_only",
            "resid_only",
            "composition_residual",
            "composition_residual_orthogonal",
        }:
            raise ValueError(f"Unknown bulk_feature_mode: {self.bulk_feature_mode}")
        if self.fraction_feature_mode not in {"raw_log", "raw_only", "log_only"}:
            raise ValueError(f"Unknown fraction_feature_mode: {self.fraction_feature_mode}")
        if self.fraction_input_scope not in {"both", "local_only", "donor_only", "none"}:
            raise ValueError(f"Unknown fraction_input_scope: {self.fraction_input_scope}")
        if self.fraction_transform_tau <= 0:
            raise ValueError("fraction_transform_tau must be positive.")
        if self.fraction_min_clip <= 0:
            raise ValueError("fraction_min_clip must be positive.")
        if self.shared_routing_mode not in {"shared_all", "exc_inh_split", "ct_specific", "ct_specific_local"}:
            raise ValueError(f"Unknown shared_routing_mode: {self.shared_routing_mode}")
        if self.fraction_gate_mode not in {"none", "raw", "sqrt", "saturating"}:
            raise ValueError(f"Unknown fraction_gate_mode: {self.fraction_gate_mode}")
        if self.tower_mode not in {"mlp", "factorized_dot", "common_plus_deviation"}:
            raise ValueError(f"Unknown tower_mode: {self.tower_mode}")
        if self.fraction_gate_mode != "none" and self.use_fraction_path_regularization:
            raise ValueError("fraction_gate_mode requires use_fraction_path_regularization=False.")
        if not 0.0 <= self.fraction_gate_floor <= 1.0:
            raise ValueError("fraction_gate_floor must be between 0 and 1.")
        if self.tower_rank <= 0:
            raise ValueError("tower_rank must be positive.")
        if self.ct_names and len(self.ct_names) != self.n_ct:
            raise ValueError(f"ct_names length {len(self.ct_names)} does not match output_dim {self.n_ct}")
        shared_group_labels, output_group_idx = self._build_shared_group_index(self.ct_names, self.shared_routing_mode)
        self.shared_group_labels = tuple(shared_group_labels)
        self.register_buffer(
            "output_group_index",
            torch.tensor(output_group_idx, dtype=torch.long),
            persistent=False,
        )
        self.n_shared_groups = len(shared_group_labels)
        self.shared_routing_uses_local_context = self.shared_routing_mode == "ct_specific_local"
        self.local_static_dim = max(0, self.local_dim - 2)
        self.register_buffer(
            "donor_fraction_mask",
            self._build_fraction_mask(self.ct_names, donor_fraction_zero_cts),
            persistent=False,
        )
        self.register_buffer(
            "local_fraction_mask",
            self._build_fraction_mask(self.ct_names, local_fraction_zero_cts),
            persistent=False,
        )

        if self.use_fraction_path_regularization:
            self.bulk_encoder = make_mlp(self.bulk_dim + gene_dim, [64], 48, dropout)
            self.fraction_encoder = make_mlp(self.n_ct * 2, [self.fraction_branch_hidden_dim], self.fraction_branch_hidden_dim, 0.0)
            if self.shared_routing_mode == "shared_all":
                self.fusion = make_mlp(48 + self.fraction_branch_hidden_dim, [48], 48, dropout)
                self.group_fusions = None
            else:
                self.fusion = None
                fusion_input_dim = 48 + self.fraction_branch_hidden_dim + (
                    self.local_static_dim if self.shared_routing_uses_local_context else 0
                )
                self.group_fusions = nn.ModuleList(
                    [make_mlp(fusion_input_dim, [48], 48, dropout) for _ in range(self.n_shared_groups)]
                )
        else:
            if self.shared_routing_mode == "shared_all":
                self.shared = make_mlp(donor_dim + gene_dim, [64], 48, dropout)
                self.group_shared = None
            else:
                self.shared = None
                shared_input_dim = donor_dim + gene_dim + (
                    self.local_static_dim if self.shared_routing_uses_local_context else 0
                )
                self.group_shared = nn.ModuleList(
                    [make_mlp(shared_input_dim, [64], 48, dropout) for _ in range(self.n_shared_groups)]
                )

        if self.tower_mode in {"mlp", "common_plus_deviation"}:
            self.ct_towers = nn.ModuleList([make_mlp(48 + local_dim, [64, 32], 1, dropout) for _ in range(output_dim)])
            if self.tower_mode == "common_plus_deviation":
                self.common_tower = make_mlp(48, [32], 1, dropout)
            else:
                self.common_tower = None
            self.ct_shared_readouts = None
            self.ct_local_readouts = None
        else:
            self.ct_towers = None
            self.common_tower = None
            self.ct_shared_readouts = nn.ModuleList([nn.Linear(48, self.tower_rank, bias=False) for _ in range(output_dim)])
            self.ct_local_readouts = nn.ModuleList([nn.Linear(local_dim, self.tower_rank, bias=False) for _ in range(output_dim)])

        if self.use_fraction_confidence:
            confidence_output_dim = self.n_ct if self.use_ct_specific_fraction_confidence else 1
            self.fraction_confidence = make_mlp(
                self.bulk_dim + self.n_ct * 2,
                [fraction_confidence_hidden_dim],
                confidence_output_dim,
                0.0,
            )
        else:
            self.fraction_confidence = None

        if self.use_fraction_refinement:
            self.fraction_refinement = make_mlp(self.bulk_dim + self.n_ct * 2, [fraction_refinement_hidden_dim], self.n_ct, 0.0)
        else:
            self.fraction_refinement = None

        if self.use_fraction_refiner:
            self.fraction_refiner = make_mlp(
                self.donor_summary_dim + self.n_ct * 2,
                repeated_hidden_dims(fraction_refiner_hidden_dim, fraction_refiner_num_layers),
                self.n_ct,
                fraction_refiner_dropout,
            )
        else:
            self.fraction_refiner = None

    def _build_fraction_mask(self, ct_names: list[str], zero_spec: str) -> torch.Tensor:
        mask = torch.ones(self.n_ct, dtype=torch.float32)
        spec = str(zero_spec).strip()
        if not spec:
            return mask
        if not ct_names:
            raise ValueError("ct_names must be provided when donor/local fraction zero celltypes are used.")
        wanted = {part.strip() for part in spec.split(",") if part.strip()}
        unknown = wanted.difference(ct_names)
        if unknown:
            raise ValueError(f"Unknown cell types in fraction zero spec {sorted(unknown)}; valid={ct_names}")
        for idx, ct in enumerate(ct_names):
            if ct in wanted:
                mask[idx] = 0.0
        return mask

    def _build_shared_group_index(self, ct_names: list[str], mode: str) -> tuple[list[str], list[int]]:
        if mode == "shared_all":
            return ["all"], [0] * self.n_ct
        if not ct_names:
            raise ValueError("ct_names must be provided when shared_routing_mode is not shared_all.")
        label_to_idx: dict[str, int] = {}
        labels: list[str] = []
        out_idx: list[int] = []
        for ct in ct_names:
            if mode in {"ct_specific", "ct_specific_local"}:
                label = ct
            elif mode == "exc_inh_split":
                label = ct if ct in {"Excitatory", "Inhibitory"} else "glia"
            else:
                raise ValueError(f"Unknown shared_routing_mode: {mode}")
            if label not in label_to_idx:
                label_to_idx[label] = len(labels)
                labels.append(label)
            out_idx.append(label_to_idx[label])
        return labels, out_idx

    def _expand_group_latents(self, group_latents: list[torch.Tensor]) -> torch.Tensor:
        if len(group_latents) == 1:
            return group_latents[0].unsqueeze(1).expand(-1, self.n_ct, -1)
        group_tensor = torch.stack(group_latents, dim=1)
        return group_tensor.index_select(1, self.output_group_index)

    def _append_group_local_context(self, base_input: torch.Tensor, local_static: torch.Tensor, group_idx: int) -> torch.Tensor:
        if not self.shared_routing_uses_local_context:
            return base_input
        return torch.cat([base_input, local_static[:, group_idx, :]], dim=1)

    def _select_bulk_features(self, donor_bulk: torch.Tensor) -> torch.Tensor:
        if self.bulk_feature_mode == "full":
            return donor_bulk
        out = donor_bulk.clone()
        if self.bulk_feature_mode == "bulk_resid":
            out[:, 1] = 0.0
        elif self.bulk_feature_mode == "bulk_only":
            out[:, 1:] = 0.0
        elif self.bulk_feature_mode in {"resid_only", "composition_residual", "composition_residual_orthogonal"}:
            out[:, :2] = 0.0
        return out

    def _select_fraction_features(self, frac_raw: torch.Tensor, frac_log: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.fraction_feature_mode == "raw_log":
            return frac_raw, frac_log
        if self.fraction_feature_mode == "raw_only":
            return frac_raw, torch.zeros_like(frac_log)
        if self.fraction_feature_mode == "log_only":
            return torch.zeros_like(frac_raw), frac_log
        raise ValueError(f"Unknown fraction_feature_mode: {self.fraction_feature_mode}")

    @staticmethod
    def _gate_fraction_pair(frac_raw: torch.Tensor, frac_log: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        if confidence.shape[1] == 1:
            return torch.cat([frac_raw, frac_log], dim=1) * confidence
        return torch.cat([frac_raw * confidence, frac_log * confidence], dim=1)

    def _fraction_residual_gate(self, frac_raw: torch.Tensor) -> torch.Tensor:
        if self.fraction_gate_mode == "none":
            return torch.ones_like(frac_raw)
        frac = torch.clamp(frac_raw, min=0.0, max=1.0)
        if self.fraction_gate_mode == "raw":
            gate = frac
        elif self.fraction_gate_mode == "sqrt":
            gate = torch.sqrt(torch.clamp(frac, min=0.0))
        elif self.fraction_gate_mode == "saturating":
            tau = max(self.fraction_gate_tau, 1e-8)
            gate = frac / (frac + tau)
        else:
            raise ValueError(f"Unknown fraction_gate_mode: {self.fraction_gate_mode}")
        if self.fraction_gate_floor > 0:
            gate = self.fraction_gate_floor + (1.0 - self.fraction_gate_floor) * gate
        return torch.clamp(gate, min=0.0, max=1.0)

    def _prepare_fraction_inputs(
        self,
        donor_feat: torch.Tensor,
        local_feat: torch.Tensor,
        donor_summary_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        donor_bulk = donor_feat[:, : self.bulk_dim]
        donor_bulk_eff = self._select_bulk_features(donor_bulk)
        frac_raw = donor_feat[:, self.bulk_dim : self.bulk_dim + self.n_ct]
        frac_log = donor_feat[:, self.bulk_dim + self.n_ct : self.bulk_dim + 2 * self.n_ct]
        local_static = local_feat[:, :, 2:]
        aux: dict[str, torch.Tensor] = {}

        if self.fraction_refiner is not None:
            if donor_summary_feat is None:
                if self.donor_summary_dim > 0:
                    raise ValueError("donor_summary_feat is required when use_fraction_refiner is enabled.")
                donor_summary_feat = donor_feat.new_zeros((donor_feat.shape[0], 0))
            refine_input = torch.cat([donor_summary_feat, frac_raw, frac_log], dim=1)
            delta = self.fraction_refiner(refine_input)
            adj_logits = torch.log(torch.clamp(frac_raw, min=self.fraction_refiner_eps)) + delta
            frac_used = torch.softmax(adj_logits, dim=1)
            aux["fraction_refine_delta"] = delta
            aux["fraction_refine_penalty"] = torch.mean((frac_used - frac_raw) ** 2)
        elif self.fraction_refinement is not None:
            refine_input = torch.cat([donor_bulk, frac_raw, frac_log], dim=1)
            delta = self.fraction_refinement(refine_input)
            adj_logits = torch.log(torch.clamp(frac_raw, min=1e-6)) + delta
            frac_used = torch.softmax(adj_logits, dim=1)
            aux["fraction_refine_delta"] = delta
            aux["fraction_refine_penalty"] = torch.mean((frac_used - frac_raw) ** 2)
        else:
            frac_used = frac_raw

        frac_log_used = transform_fraction_torch(
            frac_used,
            self.fraction_transform_mode,
            self.fraction_transform_tau,
            self.fraction_min_clip,
        )
        frac_raw_eff, frac_log_eff = self._select_fraction_features(frac_used, frac_log_used)
        if self.fraction_confidence is not None:
            conf_input = torch.cat([donor_bulk_eff, frac_raw_eff, frac_log_eff], dim=1)
            conf = torch.sigmoid(self.fraction_confidence(conf_input))
            aux["fraction_confidence_mean"] = conf.mean()
        else:
            conf = torch.ones((donor_feat.shape[0], 1), dtype=donor_feat.dtype, device=donor_feat.device)

        donor_frac_raw_eff = frac_raw_eff * self.donor_fraction_mask.view(1, -1)
        donor_frac_log_eff = frac_log_eff * self.donor_fraction_mask.view(1, -1)
        local_frac_raw_eff = frac_raw_eff * self.local_fraction_mask.view(1, -1)
        local_frac_log_eff = frac_log_eff * self.local_fraction_mask.view(1, -1)
        local_frac = torch.stack([local_frac_raw_eff, local_frac_log_eff], dim=2)
        aux["fraction_input"] = frac_raw
        aux["fraction_refined"] = frac_used
        aux["fraction_raw_effective"] = donor_frac_raw_eff
        aux["fraction_log_effective"] = donor_frac_log_eff
        aux["fraction_raw_effective_local"] = local_frac_raw_eff
        aux["fraction_log_effective_local"] = local_frac_log_eff
        return donor_bulk_eff, frac_used, local_frac, local_static, aux | {"fraction_confidence": conf}

    def forward_with_aux(
        self,
        donor_feat: torch.Tensor,
        gene_feat: torch.Tensor,
        local_feat: torch.Tensor,
        donor_summary_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        donor_bulk, frac_used, local_frac, local_static, aux = self._prepare_fraction_inputs(donor_feat, local_feat, donor_summary_feat)
        confidence = aux["fraction_confidence"]

        if self.use_fraction_path_regularization:
            bulk_repr = self.bulk_encoder(torch.cat([donor_bulk, gene_feat], dim=1))
            frac_repr_input = self._gate_fraction_pair(aux["fraction_raw_effective"], aux["fraction_log_effective"], confidence)
            if self.fraction_input_scope in {"local_only", "none"}:
                frac_repr_input = torch.zeros_like(frac_repr_input)
            frac_repr = self.fraction_encoder(frac_repr_input)
            if self.fraction_branch_scale != 1.0:
                frac_repr = frac_repr * self.fraction_branch_scale
            frac_repr = grad_scale(frac_repr, self.fraction_gradient_scale)
            if self.fraction_branch_dropout > 0:
                frac_repr = F.dropout(frac_repr, p=self.fraction_branch_dropout, training=self.training)
            fusion_in = torch.cat([bulk_repr, frac_repr], dim=1)
            if self.shared_routing_mode == "shared_all":
                shared_by_ct = self._expand_group_latents([self.fusion(fusion_in)])
            else:
                assert self.group_fusions is not None
                shared_by_ct = self._expand_group_latents(
                    [
                        mod(self._append_group_local_context(fusion_in, local_static, gi))
                        for gi, mod in enumerate(self.group_fusions)
                    ]
                )
            local_frac = local_frac * confidence.unsqueeze(2)
            if self.fraction_input_scope in {"donor_only", "none"}:
                local_frac = torch.zeros_like(local_frac)
            if self.fraction_branch_scale != 1.0:
                local_frac = local_frac * self.fraction_branch_scale
            local_eff = torch.cat([local_frac, local_static], dim=2)
        else:
            donor_frac = self._gate_fraction_pair(aux["fraction_raw_effective"], aux["fraction_log_effective"], confidence)
            if self.fraction_input_scope in {"local_only", "none"}:
                donor_frac = torch.zeros_like(donor_frac)
            donor_eff = torch.cat([donor_bulk, donor_frac], dim=1)
            local_eff = torch.cat([local_frac * confidence.unsqueeze(2), local_static], dim=2)
            if self.fraction_input_scope in {"donor_only", "none"}:
                local_eff = torch.cat([torch.zeros_like(local_frac), local_static], dim=2)
            donor_in = torch.cat([donor_eff, gene_feat], dim=1)
            if self.shared_routing_mode == "shared_all":
                shared_by_ct = self._expand_group_latents([self.shared(donor_in)])
            else:
                assert self.group_shared is not None
                shared_by_ct = self._expand_group_latents(
                    [
                        mod(self._append_group_local_context(donor_in, local_static, gi))
                        for gi, mod in enumerate(self.group_shared)
                    ]
                )

        shared = shared_by_ct.mean(dim=1)
        aux["shared_repr"] = shared
        aux["shared_repr_by_ct"] = shared_by_ct
        if self.tower_mode in {"mlp", "common_plus_deviation"}:
            assert self.ct_towers is not None
            tower_raw = torch.cat(
                [tower(torch.cat([shared_by_ct[:, ci, :], local_eff[:, ci, :]], dim=1)) for ci, tower in enumerate(self.ct_towers)],
                dim=1,
            )
            if self.tower_mode == "common_plus_deviation":
                assert self.common_tower is not None
                common = self.common_tower(shared)
                deviation = tower_raw - tower_raw.mean(dim=1, keepdim=True)
                out = common + float(self.tower_deviation_scale) * deviation
                aux["tower_common"] = common
                aux["tower_deviation_centered"] = deviation
            else:
                out = tower_raw
            aux["tower_raw"] = tower_raw
        else:
            assert self.ct_shared_readouts is not None and self.ct_local_readouts is not None
            out = torch.cat(
                [
                    (
                        self.ct_shared_readouts[ci](shared_by_ct[:, ci, :]) * self.ct_local_readouts[ci](local_eff[:, ci, :])
                    ).sum(dim=1, keepdim=True)
                    / np.sqrt(float(self.tower_rank))
                    for ci in range(self.n_ct)
                ],
                dim=1,
            )
        gate = self._fraction_residual_gate(frac_used)
        aux["fraction_residual_gate"] = gate
        if self.fraction_gate_mode != "none":
            out = out * gate
        return out, aux

    def forward(
        self,
        donor_feat: torch.Tensor,
        gene_feat: torch.Tensor,
        local_feat: torch.Tensor,
        donor_summary_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, _ = self.forward_with_aux(donor_feat, gene_feat, local_feat, donor_summary_feat)
        return out


def apply_fraction_noise(
    donor_feat: torch.Tensor,
    local_feat: torch.Tensor,
    n_ct: int,
    noise_type: str,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise_scale <= 0:
        return donor_feat, local_feat

    donor_bulk = donor_feat[:, :3]
    frac_raw = donor_feat[:, 3 : 3 + n_ct]
    local_static = local_feat[:, :, 2:]
    if noise_type == "gaussian":
        noisy = torch.clamp(frac_raw + torch.randn_like(frac_raw) * noise_scale, min=0.0)
        norm = noisy.sum(dim=1, keepdim=True).clamp(min=1e-6)
        noisy = noisy / norm
    elif noise_type == "dirichlet":
        concentration = max(1.0 / max(noise_scale * noise_scale, 1e-6), 1.0)
        alpha = torch.clamp(frac_raw, min=1e-6) * concentration + 1e-3
        noisy = torch.distributions.Dirichlet(alpha).rsample()
    else:
        raise ValueError(f"Unknown fraction_noise_type: {noise_type}")

    noisy_log = torch.log(torch.clamp(noisy, min=1e-6))
    donor_noisy = torch.cat([donor_bulk, noisy, noisy_log], dim=1)
    local_noisy = torch.cat([torch.stack([noisy, noisy_log], dim=2), local_static], dim=2)
    return donor_noisy, local_noisy


def frac_to_error_space(frac: torch.Tensor, space: str, min_clip: float) -> torch.Tensor:
    frac_clip = torch.clamp(frac, min=min_clip)
    if space == "raw":
        return frac_clip
    if space == "log":
        return torch.log(frac_clip)
    if space == "log_ratio":
        log_frac = torch.log(frac_clip)
        return log_frac - log_frac.mean(dim=1, keepdim=True)
    raise ValueError(f"Unknown fraction_error_space: {space}")


def frac_from_error_space(values: torch.Tensor, space: str, min_clip: float) -> torch.Tensor:
    if space == "raw":
        frac = torch.clamp(values, min=min_clip)
        return frac / frac.sum(dim=1, keepdim=True).clamp(min=min_clip)
    if space in {"log", "log_ratio"}:
        return torch.softmax(values, dim=1)
    raise ValueError(f"Unknown fraction_error_space: {space}")


def normalize_simplex(frac: torch.Tensor, min_clip: float) -> torch.Tensor:
    frac = torch.clamp(frac, min=min_clip)
    return frac / frac.sum(dim=1, keepdim=True).clamp(min=min_clip)


def load_fraction_error_model(path: Path | None, device: torch.device | None = None) -> dict[str, object] | None:
    if path is None or not Path(path).exists():
        return None
    with open(path) as handle:
        payload = json.load(handle)
    out: dict[str, object] = dict(payload)
    mean = torch.tensor(payload["mean"], dtype=torch.float32, device=device)
    cov = torch.tensor(payload["cov"], dtype=torch.float32, device=device)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    cov_sqrt = eigvecs @ torch.diag(torch.sqrt(eigvals))
    out["mean_t"] = mean
    out["cov_sqrt_t"] = cov_sqrt
    out["min_fraction_clip"] = float(payload.get("min_fraction_clip", 1e-6))
    out["fraction_error_space"] = str(payload.get("fraction_error_space", "log_ratio"))
    return out


def sample_fraction_perturbation(
    base_frac: torch.Tensor,
    distribution: str,
    scale: float,
    min_clip: float,
    error_model: dict[str, object] | None = None,
    add_bias: bool = False,
) -> torch.Tensor:
    if distribution == "gaussian_logfrac":
        logits = torch.log(torch.clamp(base_frac, min=min_clip))
        return torch.softmax(logits + torch.randn_like(logits) * scale, dim=1)
    if distribution == "learned_error_model":
        if error_model is None:
            raise ValueError("learned_error_model distribution requested without an error model.")
        space = str(error_model["fraction_error_space"])
        base = frac_to_error_space(base_frac, space, float(error_model["min_fraction_clip"]))
        mean = error_model["mean_t"]
        cov_sqrt = error_model["cov_sqrt_t"]
        noise = torch.randn(base.shape[0], base.shape[1], dtype=base.dtype, device=base.device) @ cov_sqrt.T
        shifted = base + noise
        if add_bias:
            shifted = shifted + mean
        return frac_from_error_space(shifted, space, float(error_model["min_fraction_clip"]))
    raise ValueError(f"Unknown perturbation_distribution: {distribution}")


def model_forward_with_aux(
    model: nn.Module,
    donor_feat: torch.Tensor,
    gene_feat: torch.Tensor,
    local_feat: torch.Tensor,
    donor_summary_feat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if hasattr(model, "forward_with_aux"):
        return model.forward_with_aux(donor_feat, gene_feat, local_feat, donor_summary_feat)
    return model(donor_feat, gene_feat, local_feat), {}


def build_model_config(args: object) -> dict[str, object]:
    return {
        "use_fraction_noise": bool(getattr(args, "use_fraction_noise", False)),
        "fraction_noise_type": str(getattr(args, "fraction_noise_type", "gaussian")),
        "fraction_noise_scale": float(getattr(args, "fraction_noise_scale", 0.0)),
        "use_fraction_path_regularization": bool(getattr(args, "use_fraction_path_regularization", False)),
        "fraction_branch_hidden_dim": int(getattr(args, "fraction_branch_hidden_dim", 16)),
        "fraction_branch_dropout": float(getattr(args, "fraction_branch_dropout", 0.0)),
        "fraction_branch_scale": float(getattr(args, "fraction_branch_scale", 1.0)),
        "fraction_gradient_scale": float(getattr(args, "fraction_gradient_scale", 1.0)),
        "fraction_gate_mode": str(getattr(args, "fraction_gate_mode", "none")),
        "fraction_gate_floor": float(getattr(args, "fraction_gate_floor", 0.0)),
        "fraction_gate_tau": float(getattr(args, "fraction_gate_tau", 0.05)),
        "tower_mode": str(getattr(args, "tower_mode", "mlp")),
        "tower_rank": int(getattr(args, "tower_rank", 16)),
        "tower_deviation_scale": float(getattr(args, "tower_deviation_scale", 1.0)),
        "residual_prior_mode": str(getattr(args, "residual_prior_mode", "none")),
        "residual_prior_multiplier": float(getattr(args, "residual_prior_multiplier", 1.0)),
        "use_fraction_confidence": bool(getattr(args, "use_fraction_confidence", False)),
        "fraction_confidence_hidden_dim": int(getattr(args, "fraction_confidence_hidden_dim", 16)),
        "use_ct_specific_fraction_confidence": bool(getattr(args, "use_ct_specific_fraction_confidence", False)),
        "bulk_feature_mode": str(getattr(args, "bulk_feature_mode", "full")),
        "fraction_orthogonal_rho": float(getattr(args, "fraction_orthogonal_rho", 1.0)),
        "target_anchor_calibration_mode": str(getattr(args, "target_anchor_calibration_mode", "none")),
        "target_residual_norm_mode": str(getattr(args, "target_residual_norm_mode", "none")),
        "composition_null_aug_lambda": float(getattr(args, "composition_null_aug_lambda", 0.0)),
        "composition_null_aug_modes": str(getattr(args, "composition_null_aug_modes", "independent,low_inh,high_inh")),
        "composition_null_aug_shift_factor": float(getattr(args, "composition_null_aug_shift_factor", 0.5)),
        "fraction_feature_mode": str(getattr(args, "fraction_feature_mode", "raw_log")),
        "fraction_input_scope": str(getattr(args, "fraction_input_scope", "both")),
        "fraction_transform_mode": str(
            getattr(args, "fraction_transform_mode", FRACTION_TRANSFORM_LEGACY_LOG)
        ),
        "fraction_transform_tau": float(getattr(args, "fraction_transform_tau", 0.01)),
        "fraction_min_clip": float(
            getattr(args, "fraction_min_clip", getattr(args, "min_fraction_clip", 1e-6))
        ),
        "donor_fraction_zero_cts": str(getattr(args, "donor_fraction_zero_cts", "")),
        "local_fraction_zero_cts": str(getattr(args, "local_fraction_zero_cts", "")),
        "shared_routing_mode": str(getattr(args, "shared_routing_mode", "shared_all")),
        "use_fraction_refinement": bool(getattr(args, "use_fraction_refinement", False)),
        "fraction_refinement_hidden_dim": int(getattr(args, "fraction_refinement_hidden_dim", 16)),
        "lambda_fraction_refine": float(getattr(args, "lambda_fraction_refine", 0.0)),
        "use_fraction_refiner": bool(getattr(args, "use_fraction_refiner", False)),
        "fraction_refiner_hidden_dim": int(getattr(args, "fraction_refiner_hidden_dim", 16)),
        "fraction_refiner_num_layers": int(getattr(args, "fraction_refiner_num_layers", 1)),
        "fraction_refiner_dropout": float(getattr(args, "fraction_refiner_dropout", 0.0)),
        "fraction_refiner_eps": float(getattr(args, "fraction_refiner_eps", 1e-6)),
        "use_noisy_fraction_prior": bool(getattr(args, "use_noisy_fraction_prior", False)),
        "prior_perturbation_distribution": str(getattr(args, "prior_perturbation_distribution", "gaussian_logfrac")),
        "prior_perturbation_scale": float(getattr(args, "prior_perturbation_scale", 0.0)),
        "lambda_frac_prior": float(getattr(args, "lambda_frac_prior", 0.0)),
        "lambda_frac_delta_l2": float(getattr(args, "lambda_frac_delta_l2", 0.0)),
        "use_bulk_consistency_loss": bool(getattr(args, "use_bulk_consistency_loss", False)),
        "lambda_bulk_consistency": float(getattr(args, "lambda_bulk_consistency", 0.0)),
        "bulk_loss_type": str(getattr(args, "bulk_loss_type", "huber")),
        "bulk_loss_delta": float(getattr(args, "bulk_loss_delta", 1.0)),
        "use_fraction_stability_loss": bool(getattr(args, "use_fraction_stability_loss", False)),
        "lambda_fraction_stability": float(getattr(args, "lambda_fraction_stability", 0.0)),
        "num_fraction_perturbations": int(getattr(args, "num_fraction_perturbations", 1)),
        "perturbation_scale": float(getattr(args, "perturbation_scale", 0.0)),
        "perturbation_distribution": str(getattr(args, "perturbation_distribution", "gaussian_logfrac")),
        "use_learned_fraction_error_model": bool(getattr(args, "use_learned_fraction_error_model", False)),
        "fraction_error_model_path": str(getattr(args, "fraction_error_model_path", "")),
        "fraction_error_space": str(getattr(args, "fraction_error_space", "log_ratio")),
        "fraction_error_shrinkage": float(getattr(args, "fraction_error_shrinkage", 0.0)),
        "min_fraction_clip": float(getattr(args, "min_fraction_clip", 1e-6)),
        "init_checkpoint": str(getattr(args, "init_checkpoint", "")),
        "reference_prior_lambda": float(getattr(args, "reference_prior_lambda", 0.0)),
        "reference_prior_penalty": str(getattr(args, "reference_prior_penalty", "l2")),
        "target_bulk_lambda": float(getattr(args, "target_bulk_lambda", 0.0)),
        "target_bulk_loss": str(getattr(args, "target_bulk_loss", "huber")),
        "target_tether_lambda": float(getattr(args, "target_tether_lambda", 0.0)),
        "target_tether_penalty": str(getattr(args, "target_tether_penalty", "huber")),
        "expression_transform": str(
            getattr(args, "expression_transform", EXPRESSION_TRANSFORM_LOG2P1_NONNEG)
        ),
        "variant_id": str(getattr(args, "variant_id", "")),
        "loss_reduction": str(getattr(args, "loss_reduction", "row_mean")),
        "subject_balanced_batch": bool(getattr(args, "subject_balanced_batch", False)),
        "subjects_per_batch": int(getattr(args, "subjects_per_batch", 16)),
        "rows_per_subject": int(getattr(args, "rows_per_subject", 0)),
    }


def scale_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        return np.ones_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def compute_donor_summary_numpy(
    bulk_log_raw: np.ndarray,
    expected_ref_log: np.ndarray,
    bulk_resid_ref: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            bulk_log_raw.mean(axis=1),
            bulk_log_raw.std(axis=1),
            expected_ref_log.mean(axis=1),
            expected_ref_log.std(axis=1),
            bulk_resid_ref.mean(axis=1),
            bulk_resid_ref.std(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def clr_numpy(frac: np.ndarray, min_clip: float = 1e-6) -> np.ndarray:
    arr = np.clip(np.asarray(frac, dtype=np.float64), min_clip, None)
    arr = arr / arr.sum(axis=1, keepdims=True)
    log_arr = np.log(arr)
    return (log_arr - log_arr.mean(axis=1, keepdims=True)).astype(np.float32)


def fraction_orthogonal_residual_numpy(resid: np.ndarray, frac: np.ndarray, rho: float = 1.0) -> np.ndarray:
    """Remove a scaled gene-wise linear association between residual bulk and CLR fractions."""

    z = clr_numpy(frac)
    design = np.concatenate([np.ones((z.shape[0], 1), dtype=np.float32), z], axis=1).astype(np.float64)
    beta, *_ = np.linalg.lstsq(design, np.asarray(resid, dtype=np.float64), rcond=None)
    fitted = design @ beta
    return (np.asarray(resid, dtype=np.float64) - float(rho) * fitted).astype(np.float32)


def calibrate_reference_anchor_numpy(
    bulk_log_raw: np.ndarray,
    frac_arr: np.ndarray,
    ref_mean_raw: np.ndarray,
    expression_transform: str,
    mode: str = "none",
    slope_min: float = 0.0,
    slope_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | str]]:
    """Build a disease-blind target-calibrated CTS anchor.

    ``gene_affine`` fits a per-gene affine map from the reference mixture to
    target bulk: B_g ~= a_g * sum_k f_k mu_kg + b_g. The fitted map is then
    applied to each cell type for that gene.
    """

    mode = str(mode)
    ref_lin = inverse_transform_expression_numpy(np.asarray(ref_mean_raw, dtype=np.float32), expression_transform)
    if mode == "none":
        return ref_lin.astype(np.float32), np.asarray(ref_mean_raw, dtype=np.float32), {"mode": mode}
    if mode != "gene_affine":
        raise ValueError(f"Unknown target_anchor_calibration_mode: {mode}")

    frac_np = np.asarray(frac_arr, dtype=np.float64)
    x = frac_np @ np.asarray(ref_lin, dtype=np.float64).T
    y = inverse_transform_expression_numpy(np.asarray(bulk_log_raw, dtype=np.float32), expression_transform).astype(np.float64)
    x_mu = x.mean(axis=0)
    y_mu = y.mean(axis=0)
    x_center = x - x_mu[None, :]
    y_center = y - y_mu[None, :]
    x_var = np.mean(x_center * x_center, axis=0)
    cov = np.mean(x_center * y_center, axis=0)
    slope = np.ones_like(x_mu, dtype=np.float64)
    ok = x_var > EPS
    slope[ok] = cov[ok] / x_var[ok]
    slope = np.clip(slope, float(slope_min), float(slope_max))
    intercept = y_mu - slope * x_mu
    calibrated_lin = slope[:, None] * np.asarray(ref_lin, dtype=np.float64) + intercept[:, None]
    calibrated_lin = np.clip(calibrated_lin, a_min=0.0, a_max=None).astype(np.float32)
    calibrated_raw = transform_expression_numpy(calibrated_lin, expression_transform).astype(np.float32)
    info: dict[str, np.ndarray | str] = {
        "mode": mode,
        "slope": slope.astype(np.float32),
        "intercept": intercept.astype(np.float32),
    }
    return calibrated_lin, calibrated_raw, info


def normalize_target_residual_numpy(
    residual: np.ndarray,
    scalers: dict[str, np.ndarray | float] | None,
    mode: str = "none",
) -> np.ndarray:
    """Disease-blind target residual normalization.

    ``target_center_scale`` z-scores the current target residual matrix and
    maps it back to the reference residual mean/SD used by downstream scaling.
    """

    mode = str(mode)
    if mode == "none":
        return np.asarray(residual, dtype=np.float32)
    if mode != "target_center_scale":
        raise ValueError(f"Unknown target_residual_norm_mode: {mode}")
    if scalers is None:
        raise ValueError("target_center_scale residual normalization requires reference scalers.")
    arr = np.asarray(residual, dtype=np.float32)
    cur_mu = float(arr.mean())
    cur_sd = float(arr.std())
    if cur_sd < EPS:
        cur_sd = 1.0
    ref_mu = float(scalers["resid_mu"])
    ref_sd = float(scalers["resid_sd"])
    if ref_sd < EPS:
        ref_sd = 1.0
    return (((arr - cur_mu) / cur_sd) * ref_sd + ref_mu).astype(np.float32)


def build_gene_meta(
    reference_cts: pd.DataFrame,
    genes: list[str],
    cts: list[str],
    expression_transform: str = EXPRESSION_TRANSFORM_LOG2P1_NONNEG,
    observation_weights: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    expression_transform = validate_expression_transform(expression_transform)
    weight_arr: np.ndarray | None = None
    if observation_weights is not None:
        missing_donors = reference_cts.index.astype(str).difference(observation_weights.index.astype(str))
        missing_cts = set(cts).difference(observation_weights.columns.astype(str))
        if len(missing_donors) or missing_cts:
            raise ValueError(
                "observation_weights do not cover reference CTS data: "
                f"missing_donors={len(missing_donors)} missing_cts={sorted(missing_cts)}"
            )
        aligned_weights = observation_weights.copy()
        aligned_weights.index = aligned_weights.index.astype(str)
        aligned_weights.columns = aligned_weights.columns.astype(str)
        weight_arr = aligned_weights.loc[reference_cts.index.astype(str), cts].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        if not np.isfinite(weight_arr).all() or (weight_arr < 0).any():
            raise ValueError("observation_weights must be finite and nonnegative.")

    ref_mean = []
    ref_resid_sd = []
    ref_ct_arrays = []
    for ci, ct in enumerate(cts):
        cols = [f"{g}_{ct}" for g in genes]
        arr = transform_expression_numpy(
            reference_cts.loc[:, cols].to_numpy(dtype=np.float32, copy=False),
            expression_transform,
        )
        ref_ct_arrays.append(arr)
        if weight_arr is None:
            ref_mean.append(arr.mean(axis=0))
            ref_resid_sd.append(arr.std(axis=0))
        else:
            weights = weight_arr[:, ci]
            weight_sum = float(weights.sum())
            if weight_sum <= EPS:
                raise ValueError(f"Cell type {ct} has zero total observation weight.")
            mean = np.sum(arr.astype(np.float64) * weights[:, None], axis=0) / weight_sum
            variance = np.sum((arr.astype(np.float64) - mean[None, :]) ** 2 * weights[:, None], axis=0) / weight_sum
            ref_mean.append(mean.astype(np.float32))
            ref_resid_sd.append(np.sqrt(np.maximum(variance, 0.0)).astype(np.float32))
    ref_mean_raw = np.stack(ref_mean, axis=1).astype(np.float32)
    ref_resid_sd_raw = np.stack(ref_resid_sd, axis=1).astype(np.float32)
    positive_sd = ref_resid_sd_raw[ref_resid_sd_raw > EPS]
    sd_floor = float(np.percentile(positive_sd, 5)) if positive_sd.size else 1.0
    sd_floor = max(sd_floor, EPS)
    ref_resid_sd_raw = np.maximum(ref_resid_sd_raw, sd_floor).astype(np.float32)

    ref_by_gene = np.stack(ref_ct_arrays, axis=2).astype(np.float32)
    ref_resid_chol_raw = np.zeros((len(genes), len(cts), len(cts)), dtype=np.float32)
    eye = np.eye(len(cts), dtype=np.float64)
    covariance_weights = None if weight_arr is None else weight_arr.min(axis=1)
    for gi in range(len(genes)):
        resid = ref_by_gene[:, gi, :].astype(np.float64)
        if covariance_weights is None:
            resid = resid - resid.mean(axis=0, keepdims=True)
            cov = (resid.T @ resid) / max(int(resid.shape[0]) - 1, 1)
        else:
            weight_sum = float(covariance_weights.sum())
            if weight_sum <= EPS:
                raise ValueError("No donors have positive joint observation weight for CTS covariance.")
            weighted_mean = np.sum(resid * covariance_weights[:, None], axis=0) / weight_sum
            resid = resid - weighted_mean[None, :]
            cov = (resid * covariance_weights[:, None]).T @ resid / weight_sum
        diag = np.diag(np.maximum(np.diag(cov), float(sd_floor) ** 2))
        # Mild diagonal shrinkage stabilizes per-gene CT covariance without
        # discarding the reference CTS correlation structure.
        cov = 0.75 * cov + 0.25 * diag
        cov = 0.5 * (cov + cov.T)
        jitter = max(float(np.trace(cov)) / max(len(cts), 1) * 1e-4, 1e-6)
        for _ in range(6):
            try:
                ref_resid_chol_raw[gi] = np.linalg.cholesky(cov + jitter * eye).astype(np.float32)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            ref_resid_chol_raw[gi] = np.linalg.cholesky(diag + jitter * eye).astype(np.float32)

    ref_mean_scaled = ref_mean_raw.copy()
    mu = ref_mean_scaled.mean(axis=0, keepdims=True)
    sd = ref_mean_scaled.std(axis=0, keepdims=True)
    sd[sd < EPS] = 1.0
    ref_mean_scaled = (ref_mean_scaled - mu) / sd

    top_onehot = np.zeros((len(genes), len(cts)), dtype=np.float32)
    top_idx = ref_mean_raw.argmax(axis=1)
    top_onehot[np.arange(len(genes)), top_idx] = 1.0
    top_margin = np.partition(ref_mean_raw, -2, axis=1)[:, -1] - np.partition(ref_mean_raw, -2, axis=1)[:, -2]
    ref_ct_std = ref_mean_raw.std(axis=1).astype(np.float32)
    ref_mean_abundance = ref_mean_raw.mean(axis=1).astype(np.float32)
    cont = np.stack([top_margin.astype(np.float32), ref_ct_std, ref_mean_abundance], axis=1)
    cont_mu = cont.mean(axis=0, keepdims=True)
    cont_sd = cont.std(axis=0, keepdims=True)
    cont_sd[cont_sd < EPS] = 1.0
    cont_scaled = (cont - cont_mu) / cont_sd

    gene_feature_matrix = np.concatenate([ref_mean_scaled, top_onehot, cont_scaled.astype(np.float32)], axis=1).astype(np.float32)
    local_static = np.stack([ref_mean_scaled, top_onehot], axis=2).astype(np.float32)
    return {
        "genes": np.asarray(genes, dtype=object),
        "cts": np.asarray(cts, dtype=object),
        "ref_mean_raw": ref_mean_raw.astype(np.float32),
        "ref_resid_sd_raw": ref_resid_sd_raw.astype(np.float32),
        "ref_resid_chol_raw": ref_resid_chol_raw.astype(np.float32),
        "ref_mean_scaled": ref_mean_scaled.astype(np.float32),
        "top_onehot": top_onehot.astype(np.float32),
        "gene_feature_matrix": gene_feature_matrix,
        "local_static": local_static,
        "margin_score": scale_weights(top_margin),
    }


def build_bundle(
    pbs: pd.DataFrame,
    frac: pd.DataFrame,
    truth_cts_raw: pd.DataFrame | None,
    gene_meta: dict[str, np.ndarray],
    scalers: dict[str, np.ndarray | float] | None,
    fit_scalers: bool,
    device: torch.device,
    expression_transform: str = EXPRESSION_TRANSFORM_LOG2P1_NONNEG,
    bulk_feature_mode: str = "full",
    fraction_orthogonal_rho: float = 1.0,
    target_anchor_calibration_mode: str = "none",
    target_residual_norm_mode: str = "none",
    fraction_transform_mode: str = FRACTION_TRANSFORM_LEGACY_LOG,
    fraction_transform_tau: float = 0.01,
    fraction_min_clip: float = 1e-6,
    observation_weights: pd.DataFrame | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray | float]]:
    expression_transform = validate_expression_transform(expression_transform)
    bulk_feature_mode = str(bulk_feature_mode)
    genes = list(gene_meta["genes"])
    cts = list(gene_meta["cts"])
    donors = pbs.index.astype(str).tolist()
    D = len(donors)
    G = len(genes)
    C = len(cts)

    bulk_log_raw = transform_expression_numpy(
        pbs.loc[:, genes].to_numpy(dtype=np.float32, copy=False),
        expression_transform,
    )
    frac_arr = frac.loc[:, cts].to_numpy(dtype=np.float32, copy=False)
    frac_log = transform_fraction_numpy(
        frac_arr,
        fraction_transform_mode,
        fraction_transform_tau,
        fraction_min_clip,
    )
    ref_mean_raw = np.asarray(gene_meta["ref_mean_raw"], dtype=np.float32)
    ref_mean_scaled = np.asarray(gene_meta["ref_mean_scaled"], dtype=np.float32)
    top_onehot = np.asarray(gene_meta["top_onehot"], dtype=np.float32)
    ref_lin, ref_mean_anchor_raw, anchor_calibration_info = calibrate_reference_anchor_numpy(
        bulk_log_raw,
        frac_arr,
        ref_mean_raw,
        expression_transform,
        mode=target_anchor_calibration_mode,
    )
    expected_ref_lin = frac_arr @ ref_lin.T
    expected_ref_log = transform_expression_numpy(expected_ref_lin, expression_transform)
    bulk_resid_ref = (bulk_log_raw - expected_ref_log).astype(np.float32)
    if bulk_feature_mode == "composition_residual_orthogonal":
        bulk_resid_feature = fraction_orthogonal_residual_numpy(
            bulk_resid_ref,
            frac_arr,
            rho=fraction_orthogonal_rho,
        )
    else:
        bulk_resid_feature = bulk_resid_ref
    if not fit_scalers:
        bulk_resid_feature = normalize_target_residual_numpy(
            bulk_resid_feature,
            scalers,
            mode=target_residual_norm_mode,
        )
    donor_summary_raw = compute_donor_summary_numpy(bulk_log_raw, expected_ref_log, bulk_resid_ref)

    if fit_scalers:
        bulk_mu = bulk_log_raw.mean(axis=0)
        bulk_sd = bulk_log_raw.std(axis=0)
        bulk_sd[bulk_sd < EPS] = 1.0
        expected_mu = float(expected_ref_log.mean())
        expected_sd = float(expected_ref_log.std())
        if expected_sd < EPS:
            expected_sd = 1.0
        resid_mu = float(bulk_resid_feature.mean())
        resid_sd = float(bulk_resid_feature.std())
        if resid_sd < EPS:
            resid_sd = 1.0
        donor_summary_mu = donor_summary_raw.mean(axis=0)
        donor_summary_sd = donor_summary_raw.std(axis=0)
        donor_summary_sd[donor_summary_sd < EPS] = 1.0
        scalers = {
            "bulk_mu": bulk_mu.astype(np.float32),
            "bulk_sd": bulk_sd.astype(np.float32),
            "expected_mu": expected_mu,
            "expected_sd": expected_sd,
            "resid_mu": resid_mu,
            "resid_sd": resid_sd,
            "donor_summary_mu": donor_summary_mu.astype(np.float32),
            "donor_summary_sd": donor_summary_sd.astype(np.float32),
        }
    assert scalers is not None

    bulk_scaled = ((bulk_log_raw - np.asarray(scalers["bulk_mu"])) / np.asarray(scalers["bulk_sd"])).astype(np.float32)
    expected_scaled = ((expected_ref_log - float(scalers["expected_mu"])) / float(scalers["expected_sd"])).astype(np.float32)
    resid_scaled = ((bulk_resid_feature - float(scalers["resid_mu"])) / float(scalers["resid_sd"])).astype(np.float32)
    donor_summary_scaled = (
        (donor_summary_raw - np.asarray(scalers["donor_summary_mu"])) / np.asarray(scalers["donor_summary_sd"])
    ).astype(np.float32)

    donor_features = np.concatenate(
        [
            bulk_scaled.reshape(-1, 1),
            expected_scaled.reshape(-1, 1),
            resid_scaled.reshape(-1, 1),
            np.repeat(frac_arr, G, axis=0),
            np.repeat(frac_log, G, axis=0),
        ],
        axis=1,
    ).astype(np.float32)
    gene_features = np.tile(np.asarray(gene_meta["gene_feature_matrix"], dtype=np.float32)[None, :, :], (D, 1, 1)).reshape(D * G, -1)
    frac_expanded = np.repeat(frac_arr[:, None, :], G, axis=1)
    frac_log_expanded = np.repeat(frac_log[:, None, :], G, axis=1)
    ref_scaled_expanded = np.repeat(ref_mean_scaled[None, :, :], D, axis=0)
    top_expanded = np.repeat(top_onehot[None, :, :], D, axis=0)
    local_features = np.stack([frac_expanded, frac_log_expanded, ref_scaled_expanded, top_expanded], axis=3).reshape(D * G, C, 4)
    donor_summary_rows = np.repeat(donor_summary_scaled[:, None, :], G, axis=1).reshape(D * G, donor_summary_scaled.shape[1]).astype(np.float32)
    row_donor_idx = np.repeat(np.arange(D, dtype=np.int64), G)
    row_gene_idx = np.tile(np.arange(G, dtype=np.int64), D)

    # Release adapter: inference does not require or fabricate CTS labels.
    truth_df = None
    y_abs = None
    if truth_cts_raw is not None:
        truth_cols = [f"{g}_{ct}" for g in genes for ct in cts]
        truth_df = truth_cts_raw.loc[donors, truth_cols].copy()
        truth_arr = np.zeros((D, G, C), dtype=np.float32)
        for ci, ct in enumerate(cts):
            cols = [f"{g}_{ct}" for g in genes]
            truth_arr[:, :, ci] = transform_expression_numpy(
                truth_df.loc[:, cols].to_numpy(dtype=np.float32, copy=False),
                expression_transform,
            )
        y_abs = truth_arr.reshape(D * G, C)
    ref_mean_flat = np.tile(ref_mean_anchor_raw[None, :, :], (D, 1, 1)).reshape(D * G, C).astype(np.float32)
    observation_weight_df = None
    observation_weight_rows = None
    if observation_weights is not None:
        weights = observation_weights.copy()
        weights.index = weights.index.astype(str)
        weights.columns = weights.columns.astype(str)
        missing_donors = set(donors).difference(weights.index)
        missing_cts = set(cts).difference(weights.columns)
        if missing_donors or missing_cts:
            raise ValueError(
                "observation_weights do not cover bundle data: "
                f"missing_donors={len(missing_donors)} missing_cts={sorted(missing_cts)}"
            )
        observation_weight_df = weights.loc[donors, cts].astype(np.float32)
        obs_arr = observation_weight_df.to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(obs_arr).all() or (obs_arr < 0).any():
            raise ValueError("observation_weights must be finite and nonnegative.")
        observation_weight_rows = np.repeat(obs_arr[:, None, :], G, axis=1).reshape(D * G, C)

    return {
        "donor_feat": torch.from_numpy(donor_features).to(device),
        "gene_feat": torch.from_numpy(gene_features).to(device),
        "local_feat": torch.from_numpy(local_features).to(device),
        "donor_summary_feat": torch.from_numpy(donor_summary_rows).to(device),
        "Y_abs": None if y_abs is None else torch.from_numpy(y_abs).to(device),
        "ref_mean": torch.from_numpy(ref_mean_flat).to(device),
        "observation_weight": (
            None if observation_weight_rows is None else torch.from_numpy(observation_weight_rows).to(device)
        ),
        "observation_weight_df": observation_weight_df,
        "row_donor_idx": torch.from_numpy(row_donor_idx).to(device),
        "row_gene_idx": torch.from_numpy(row_gene_idx).to(device),
        "bulk_log_matrix": torch.from_numpy(bulk_log_raw.astype(np.float32)).to(device),
        "bulk_scaled_matrix": torch.from_numpy(bulk_scaled.astype(np.float32)).to(device),
        "bulk_mu": torch.from_numpy(np.asarray(scalers["bulk_mu"], dtype=np.float32)).to(device),
        "bulk_sd": torch.from_numpy(np.asarray(scalers["bulk_sd"], dtype=np.float32)).to(device),
        "frac_matrix": torch.from_numpy(frac_arr.astype(np.float32)).to(device),
        "local_static_matrix": torch.from_numpy(np.asarray(gene_meta["local_static"], dtype=np.float32)).to(device),
        "ref_resid_sd_matrix": torch.from_numpy(np.asarray(gene_meta["ref_resid_sd_raw"], dtype=np.float32)).to(device),
        "ref_resid_chol_matrix": torch.from_numpy(np.asarray(gene_meta["ref_resid_chol_raw"], dtype=np.float32)).to(device),
        "ref_lin_matrix": torch.from_numpy(ref_lin.astype(np.float32)).to(device),
        "target_anchor_calibration_info": anchor_calibration_info,
        "donor_summary_mu": torch.from_numpy(np.asarray(scalers["donor_summary_mu"], dtype=np.float32)).to(device),
        "donor_summary_sd": torch.from_numpy(np.asarray(scalers["donor_summary_sd"], dtype=np.float32)).to(device),
        "expected_mu": torch.tensor(float(scalers["expected_mu"]), dtype=torch.float32, device=device),
        "expected_sd": torch.tensor(float(scalers["expected_sd"]), dtype=torch.float32, device=device),
        "resid_mu": torch.tensor(float(scalers["resid_mu"]), dtype=torch.float32, device=device),
        "resid_sd": torch.tensor(float(scalers["resid_sd"]), dtype=torch.float32, device=device),
        "donors": donors,
        "genes": genes,
        "cts": cts,
        "shapes": (D, G, C),
        "donor_summary_dim": int(donor_summary_scaled.shape[1]),
        "truth_raw_df": truth_df,
        "expression_transform": expression_transform,
        "fraction_transform_mode": validate_fraction_transform(fraction_transform_mode),
        "fraction_transform_tau": float(fraction_transform_tau),
        "fraction_min_clip": float(fraction_min_clip),
    }, scalers


def _compute_donor_summary_torch(
    bundle: dict[str, object],
    donor_ids: torch.Tensor,
    frac_unique: torch.Tensor,
) -> torch.Tensor:
    bulk_log = bundle["bulk_log_matrix"].index_select(0, donor_ids)
    ref_lin = bundle["ref_lin_matrix"]
    expected_ref_lin = frac_unique @ ref_lin.T
    expected_ref_log = transform_expression_torch(expected_ref_lin, str(bundle["expression_transform"]))
    resid = bulk_log - expected_ref_log
    raw = torch.stack(
        [
            bulk_log.mean(dim=1),
            bulk_log.std(dim=1, unbiased=False),
            expected_ref_log.mean(dim=1),
            expected_ref_log.std(dim=1, unbiased=False),
            resid.mean(dim=1),
            resid.std(dim=1, unbiased=False),
        ],
        dim=1,
    )
    return (raw - bundle["donor_summary_mu"]) / bundle["donor_summary_sd"]


def materialize_batch_features(
    bundle: dict[str, object],
    row_donor_idx: torch.Tensor,
    row_gene_idx: torch.Tensor,
    frac_unique: torch.Tensor | None = None,
    donor_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if donor_ids is None:
        donor_ids, inverse = torch.unique(row_donor_idx, sorted=False, return_inverse=True)
    else:
        mapping = {int(d.item()): i for i, d in enumerate(donor_ids)}
        inverse = torch.tensor([mapping[int(d.item())] for d in row_donor_idx], dtype=torch.long, device=row_donor_idx.device)
    if frac_unique is None:
        frac_unique = bundle["frac_matrix"].index_select(0, donor_ids)
    frac_rows = frac_unique.index_select(0, inverse)
    frac_log_rows = torch.log(torch.clamp(frac_rows, min=1e-6))

    bulk_scaled = bundle["bulk_scaled_matrix"][row_donor_idx, row_gene_idx].unsqueeze(1)
    bulk_log = bundle["bulk_log_matrix"][row_donor_idx, row_gene_idx]
    ref_lin_rows = bundle["ref_lin_matrix"].index_select(0, row_gene_idx)
    expected_ref_lin = (frac_rows * ref_lin_rows).sum(dim=1)
    expected_ref_log = transform_expression_torch(expected_ref_lin, str(bundle["expression_transform"]))
    expected_scaled = ((expected_ref_log - bundle["expected_mu"]) / bundle["expected_sd"]).unsqueeze(1)
    resid_scaled = (((bulk_log - expected_ref_log) - bundle["resid_mu"]) / bundle["resid_sd"]).unsqueeze(1)
    donor_feat = torch.cat([bulk_scaled, expected_scaled, resid_scaled, frac_rows, frac_log_rows], dim=1)

    local_static = bundle["local_static_matrix"].index_select(0, row_gene_idx)
    local_frac = torch.stack([frac_rows, frac_log_rows], dim=2)
    local_feat = torch.cat([local_frac, local_static], dim=2)
    donor_summary = _compute_donor_summary_torch(bundle, donor_ids, frac_unique).index_select(0, inverse)
    meta = {
        "bulk_log_rows": bulk_log,
        "frac_rows": frac_rows,
        "donor_ids": donor_ids,
        "inverse_index": inverse,
    }
    return donor_feat, local_feat, donor_summary, meta


def build_pair_indices(n_ct: int) -> tuple[torch.Tensor, torch.Tensor]:
    left = []
    right = []
    for i in range(n_ct):
        for j in range(i + 1, n_ct):
            left.append(i)
            right.append(j)
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)


def forward_pred(
    model: nn.Module,
    donor_feat: torch.Tensor,
    gene_feat: torch.Tensor,
    local_feat: torch.Tensor,
    ref_mean: torch.Tensor,
    donor_summary_feat: torch.Tensor | None = None,
    residual_prior_scale: torch.Tensor | None = None,
    residual_prior_mode: str = "none",
    residual_prior_multiplier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    delta_raw, aux = model_forward_with_aux(model, donor_feat, gene_feat, local_feat, donor_summary_feat)
    delta_hat = apply_residual_prior_transform(
        delta_raw,
        residual_prior_scale,
        residual_prior_mode,
        residual_prior_multiplier,
    )
    aux["delta_raw"] = delta_raw
    pred_abs = ref_mean + delta_hat
    return delta_hat, pred_abs, aux


def apply_residual_prior_transform(
    delta_raw: torch.Tensor,
    residual_prior_scale: torch.Tensor | None,
    residual_prior_mode: str = "none",
    residual_prior_multiplier: float = 1.0,
) -> torch.Tensor:
    mode = str(residual_prior_mode)
    if mode == "none":
        return delta_raw
    if residual_prior_scale is None:
        raise ValueError(f"residual_prior_mode={mode} requires residual_prior_scale.")
    prior = residual_prior_scale.to(device=delta_raw.device, dtype=delta_raw.dtype)
    if mode in {"ref_sd_scale", "ref_sd_tanh"}:
        scale = torch.clamp(prior, min=EPS) * float(residual_prior_multiplier)
    elif mode in {"ref_cov", "ref_cov_tanh"}:
        if prior.ndim != 3:
            raise ValueError(f"{mode} expects residual_prior_scale shape [N, CT, CT], got {tuple(prior.shape)}")
        raw = torch.tanh(delta_raw) if mode == "ref_cov_tanh" else delta_raw
        return torch.bmm(prior, raw.unsqueeze(2)).squeeze(2) * float(residual_prior_multiplier)
    else:
        raise ValueError(f"Unknown residual_prior_mode: {mode}")
    if mode == "ref_sd_scale":
        return delta_raw * scale
    if mode == "ref_sd_tanh":
        return torch.tanh(delta_raw) * scale
    raise ValueError(f"Unknown residual_prior_mode: {mode}")


def select_residual_prior_tensor(
    bundle: dict[str, object],
    row_gene_idx: torch.Tensor,
    residual_prior_mode: str = "none",
) -> torch.Tensor | None:
    mode = str(residual_prior_mode)
    if mode == "none":
        return None
    if mode in {"ref_sd_scale", "ref_sd_tanh"}:
        prior = bundle.get("ref_resid_sd_matrix")
        if prior is None:
            raise ValueError(f"{mode} requires ref_resid_sd_matrix in bundle.")
        return prior.index_select(0, row_gene_idx)
    if mode in {"ref_cov", "ref_cov_tanh"}:
        prior = bundle.get("ref_resid_chol_matrix")
        if prior is None:
            raise ValueError(f"{mode} requires ref_resid_chol_matrix in bundle.")
        return prior.index_select(0, row_gene_idx)
    raise ValueError(f"Unknown residual_prior_mode: {mode}")


def compute_fraction_prior_losses(
    aux: dict[str, torch.Tensor],
    lambda_frac_prior: float,
    lambda_frac_delta_l2: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    zero = aux["fraction_refined"].new_tensor(0.0)
    loss_prior = zero
    loss_delta = zero
    if lambda_frac_prior > 0:
        q = torch.clamp(aux["fraction_input"], min=1e-6)
        r = torch.clamp(aux["fraction_refined"], min=1e-6)
        loss_prior = F.kl_div(torch.log(r), q, reduction="batchmean")
    if lambda_frac_delta_l2 > 0 and "fraction_refine_delta" in aux:
        loss_delta = torch.mean(aux["fraction_refine_delta"] ** 2)
    total = lambda_frac_prior * loss_prior + lambda_frac_delta_l2 * loss_delta
    return total, {
        "loss_fraction_prior": float(loss_prior.detach().cpu()),
        "loss_fraction_delta_l2": float(loss_delta.detach().cpu()),
    }


def compute_bulk_consistency_loss(
    pred_abs: torch.Tensor,
    frac_rows: torch.Tensor,
    bulk_log_rows: torch.Tensor,
    loss_type: str,
    huber_delta: float,
    expression_transform: str = EXPRESSION_TRANSFORM_LOG2P1_NONNEG,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_lin = inverse_transform_expression_torch(pred_abs, expression_transform)
    recon_log = transform_expression_torch(torch.sum(frac_rows * pred_lin, dim=1), expression_transform)
    if loss_type == "mse":
        loss = torch.mean((recon_log - bulk_log_rows) ** 2)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(recon_log, bulk_log_rows, beta=huber_delta)
    else:
        raise ValueError(f"Unknown bulk_loss_type: {loss_type}")
    mae = torch.mean(torch.abs(recon_log - bulk_log_rows))
    return loss, {
        "bulk_recon_mae": float(mae.detach().cpu()),
        "bulk_recon_mse": float(torch.mean((recon_log - bulk_log_rows) ** 2).detach().cpu()),
    }


def compute_reference_prior_loss(
    pred_abs: torch.Tensor,
    ref_mean: torch.Tensor,
    penalty: str,
    huber_delta: float,
) -> torch.Tensor:
    penalty = str(penalty)
    if penalty == "l2":
        return torch.mean((pred_abs - ref_mean) ** 2)
    if penalty == "l1":
        return torch.mean(torch.abs(pred_abs - ref_mean))
    if penalty == "huber":
        return F.smooth_l1_loss(pred_abs, ref_mean, beta=huber_delta)
    raise ValueError(f"Unknown reference_prior penalty: {penalty}")


def compute_prediction_tether_loss(
    pred_abs: torch.Tensor,
    teacher_pred_abs: torch.Tensor,
    penalty: str,
    huber_delta: float,
) -> torch.Tensor:
    penalty = str(penalty)
    teacher = teacher_pred_abs.detach()
    if penalty == "l2":
        return torch.mean((pred_abs - teacher) ** 2)
    if penalty == "l1":
        return torch.mean(torch.abs(pred_abs - teacher))
    if penalty == "huber":
        return F.smooth_l1_loss(pred_abs, teacher, beta=huber_delta)
    raise ValueError(f"Unknown target_tether penalty: {penalty}")


def compute_stability_loss(
    pred_abs: torch.Tensor,
    pred_abs_perturbed: torch.Tensor,
    huber_delta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.smooth_l1_loss(pred_abs_perturbed, pred_abs.detach(), beta=huber_delta)
    mean_shift = torch.mean(torch.abs(pred_abs_perturbed - pred_abs.detach()))
    return loss, {"stability_mean_abs_change": float(mean_shift.detach().cpu())}


def _subject_mean(row_loss: torch.Tensor, row_donor_idx: torch.Tensor | None) -> torch.Tensor:
    if row_donor_idx is None:
        return row_loss.mean()
    donors, inverse = torch.unique(row_donor_idx, sorted=False, return_inverse=True)
    sums = torch.zeros(donors.shape[0], dtype=row_loss.dtype, device=row_loss.device)
    counts = torch.zeros(donors.shape[0], dtype=row_loss.dtype, device=row_loss.device)
    sums.index_add_(0, inverse, row_loss)
    counts.index_add_(0, inverse, torch.ones_like(row_loss))
    return (sums / counts.clamp(min=1.0)).mean()


def compute_ct_sep_loss(
    pred_abs: torch.Tensor,
    row_donor_idx: torch.Tensor,
    ct_similarity_targets: torch.Tensor | None,
    margin_sep: float = 0.05,
    min_donors: int = 4,
) -> tuple[torch.Tensor, int]:
    if ct_similarity_targets is None:
        return pred_abs.new_tensor(0.0), 0
    donors, inverse = torch.unique(row_donor_idx, sorted=False, return_inverse=True)
    if int(donors.numel()) < int(min_donors):
        return pred_abs.new_tensor(0.0), 0
    donor_medians = []
    for donor_idx in range(int(donors.numel())):
        mask = inverse == donor_idx
        if int(mask.sum().item()) == 0:
            continue
        donor_medians.append(pred_abs[mask].median(dim=0).values)
    if len(donor_medians) < int(min_donors):
        return pred_abs.new_tensor(0.0), 0
    med = torch.stack(donor_medians, dim=0)
    targets = ct_similarity_targets.to(device=pred_abs.device, dtype=pred_abs.dtype)
    losses = []
    for i in range(med.shape[1]):
        for j in range(i + 1, med.shape[1]):
            target = targets[i, j]
            if not torch.isfinite(target):
                continue
            x = med[:, i]
            y = med[:, j]
            x_dev = x - x.mean()
            y_dev = y - y.mean()
            denom = torch.sqrt((x_dev.pow(2).sum()) * (y_dev.pow(2).sum()))
            if float(denom.detach().cpu()) <= EPS:
                continue
            corr = (x_dev * y_dev).sum() / denom
            excess = corr - target - float(margin_sep)
            losses.append(torch.relu(excess).pow(2))
    if not losses:
        return pred_abs.new_tensor(0.0), 0
    return torch.stack(losses).mean(), len(losses)


def compute_decorr_struct_loss(
    pred_abs: torch.Tensor,
    c_ref: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pred_abs.ndim != 2:
        raise ValueError(f"compute_decorr_struct_loss expects [N, C], got {tuple(pred_abs.shape)}")
    n_rows, n_ct = pred_abs.shape
    if n_rows < 2 or n_ct < 2:
        zero = pred_abs.new_tensor(0.0)
        return zero, zero, zero
    centered = pred_abs - pred_abs.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    standardized = centered / scale
    corr_like = (standardized.transpose(0, 1) @ standardized) / float(n_rows)
    eye = torch.eye(n_ct, device=pred_abs.device, dtype=torch.bool)
    offdiag = ~eye
    offdiag_vals = corr_like[offdiag]
    if offdiag_vals.numel() == 0:
        zero = pred_abs.new_tensor(0.0)
        return zero, zero, zero
    decorr_loss = offdiag_vals.pow(2).mean()
    offdiag_mean = offdiag_vals.mean()
    if c_ref is None:
        struct_loss = pred_abs.new_tensor(0.0)
    else:
        ref = c_ref.to(device=pred_abs.device, dtype=pred_abs.dtype)
        if tuple(ref.shape) != tuple(corr_like.shape):
            raise ValueError(f"ct structure reference shape {tuple(ref.shape)} != pred corr shape {tuple(corr_like.shape)}")
        ref_offdiag = ref[offdiag]
        finite_mask = torch.isfinite(ref_offdiag)
        if int(finite_mask.sum().item()) == 0:
            struct_loss = pred_abs.new_tensor(0.0)
        else:
            struct_loss = (offdiag_vals[finite_mask] - ref_offdiag[finite_mask]).pow(2).mean()
    return decorr_loss, struct_loss, offdiag_mean


def compute_shared_latent_decorr_loss(
    shared_by_ct: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    if shared_by_ct.ndim != 3:
        raise ValueError(f"compute_shared_latent_decorr_loss expects [N, CT, H], got {tuple(shared_by_ct.shape)}")
    n_rows, n_ct, hidden = shared_by_ct.shape
    zero = shared_by_ct.new_tensor(0.0)
    if n_rows < 2 or n_ct < 2 or hidden < 1:
        return zero, {
            "shared_latent_cross_abs_mean": 0.0,
            "shared_latent_cross_abs_max": 0.0,
        }
    centered = shared_by_ct - shared_by_ct.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    standardized = centered / scale
    pair_losses = []
    pair_abs_means = []
    pair_abs_max = []
    for i in range(n_ct):
        zi = standardized[:, i, :]
        for j in range(i + 1, n_ct):
            zj = standardized[:, j, :]
            cross = (zi.transpose(0, 1) @ zj) / float(n_rows)
            abs_cross = cross.abs()
            pair_losses.append(cross.pow(2).mean())
            pair_abs_means.append(abs_cross.mean())
            pair_abs_max.append(abs_cross.max())
    if not pair_losses:
        return zero, {
            "shared_latent_cross_abs_mean": 0.0,
            "shared_latent_cross_abs_max": 0.0,
        }
    return torch.stack(pair_losses).mean(), {
        "shared_latent_cross_abs_mean": float(torch.stack(pair_abs_means).mean().detach().cpu()),
        "shared_latent_cross_abs_max": float(torch.stack(pair_abs_max).max().detach().cpu()),
    }


def compute_losses(
    delta_hat: torch.Tensor,
    pred_abs: torch.Tensor,
    y_abs: torch.Tensor,
    ref_mean: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    lambda_contrast: float,
    residual_loss_weight: float,
    huber_delta: float,
    row_donor_idx: torch.Tensor | None = None,
    loss_reduction: str = "row_mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    delta_true = y_abs - ref_mean
    pair_pred = pred_abs.index_select(1, pair_i) - pred_abs.index_select(1, pair_j)
    pair_true = y_abs.index_select(1, pair_i) - y_abs.index_select(1, pair_j)
    reduction = str(loss_reduction)
    if reduction == "row_mean":
        loss_resid = F.smooth_l1_loss(delta_hat, delta_true, beta=huber_delta)
        loss_contrast = F.smooth_l1_loss(pair_pred, pair_true, beta=huber_delta)
    elif reduction == "subject_mean":
        resid_rows = F.smooth_l1_loss(delta_hat, delta_true, beta=huber_delta, reduction="none").mean(dim=1)
        contrast_rows = F.smooth_l1_loss(pair_pred, pair_true, beta=huber_delta, reduction="none").mean(dim=1)
        loss_resid = _subject_mean(resid_rows, row_donor_idx)
        loss_contrast = _subject_mean(contrast_rows, row_donor_idx)
    else:
        raise ValueError(f"Unknown loss_reduction: {loss_reduction}")
    total = float(residual_loss_weight) * loss_resid + lambda_contrast * loss_contrast
    return total, {"loss_resid": loss_resid, "loss_contrast": loss_contrast}


def compute_weighted_losses(
    delta_hat: torch.Tensor,
    pred_abs: torch.Tensor,
    y_abs: torch.Tensor,
    ref_mean: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    lambda_contrast: float,
    residual_loss_weight: float,
    huber_delta: float,
    gene_ct_weight: torch.Tensor | None,
    row_donor_idx: torch.Tensor | None = None,
    loss_reduction: str = "row_mean",
    observation_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if gene_ct_weight is None and observation_weight is None:
        return compute_losses(
            delta_hat,
            pred_abs,
            y_abs,
            ref_mean,
            pair_i,
            pair_j,
            lambda_contrast,
            residual_loss_weight,
            huber_delta,
            row_donor_idx=row_donor_idx,
            loss_reduction=loss_reduction,
        )

    cell_weight = torch.ones_like(y_abs)
    if gene_ct_weight is not None:
        cell_weight = cell_weight * gene_ct_weight
    if observation_weight is not None:
        if observation_weight.shape != y_abs.shape:
            raise ValueError(
                f"observation_weight shape {tuple(observation_weight.shape)} does not match labels {tuple(y_abs.shape)}"
            )
        cell_weight = cell_weight * observation_weight

    delta_true = y_abs - ref_mean
    resid_raw = F.smooth_l1_loss(delta_hat, delta_true, beta=huber_delta, reduction="none")
    pair_pred = pred_abs.index_select(1, pair_i) - pred_abs.index_select(1, pair_j)
    pair_true = y_abs.index_select(1, pair_i) - y_abs.index_select(1, pair_j)
    pair_weight = torch.ones_like(pair_pred)
    if gene_ct_weight is not None:
        pair_weight = pair_weight * 0.5 * (
            gene_ct_weight.index_select(1, pair_i) + gene_ct_weight.index_select(1, pair_j)
        )
    if observation_weight is not None:
        # A contrast is observed only when both constituent CTS labels are reliable.
        pair_weight = pair_weight * torch.minimum(
            observation_weight.index_select(1, pair_i),
            observation_weight.index_select(1, pair_j),
        )
    contrast_raw = F.smooth_l1_loss(pair_pred, pair_true, beta=huber_delta, reduction="none")

    reduction = str(loss_reduction)
    if reduction == "row_mean":
        loss_resid = (resid_raw * cell_weight).sum() / cell_weight.sum().clamp(min=EPS)
        loss_contrast = (contrast_raw * pair_weight).sum() / pair_weight.sum().clamp(min=EPS)
    elif reduction == "subject_mean":
        resid_rows = (resid_raw * cell_weight).sum(dim=1) / cell_weight.sum(dim=1).clamp(min=EPS)
        contrast_rows = (contrast_raw * pair_weight).sum(dim=1) / pair_weight.sum(dim=1).clamp(min=EPS)
        loss_resid = _subject_mean(resid_rows, row_donor_idx)
        loss_contrast = _subject_mean(contrast_rows, row_donor_idx)
    else:
        raise ValueError(f"Unknown loss_reduction: {loss_reduction}")
    total = float(residual_loss_weight) * loss_resid + lambda_contrast * loss_contrast
    return total, {"loss_resid": loss_resid, "loss_contrast": loss_contrast}


def _gene_ct_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"gene", "cell_type"}.issubset(out.columns):
        out["gene"] = out["gene"].astype(str)
        out["cell_type"] = out["cell_type"].astype(str)
        return out
    if "feature" in out.columns:
        parts = out["feature"].astype(str).str.rsplit("_", n=1, expand=True)
        out["gene"] = parts[0].astype(str)
        out["cell_type"] = parts[1].astype(str)
        return out
    if "gene_celltype" in out.columns:
        parts = out["gene_celltype"].astype(str).str.rsplit("_", n=1, expand=True)
        out["gene"] = parts[0].astype(str)
        out["cell_type"] = parts[1].astype(str)
        return out
    if "gene" in out.columns:
        out["gene"] = out["gene"].astype(str)
        out["cell_type"] = ""
        return out
    raise ValueError("Gene-CT table must contain gene/cell_type, feature, gene_celltype, or gene columns.")


def load_gene_ct_weight_matrix(
    weight_file: Path | None,
    genes: list[str],
    cts: list[str],
    device: torch.device,
    column: str = "disease_blind_confidence",
    min_weight: float = 0.0,
    missing: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor | None:
    if weight_file is None:
        return None
    if not weight_file.exists():
        raise FileNotFoundError(f"gene_ct_weight_file does not exist: {weight_file}")
    frame = _gene_ct_from_frame(pd.read_csv(weight_file, sep="\t"))
    if column not in frame.columns:
        raise ValueError(f"gene_ct_weight_column={column!r} not found in {weight_file}")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")

    weights = pd.DataFrame(float(missing), index=pd.Index(genes, name="gene"), columns=cts, dtype=np.float32)
    if set(frame["cell_type"].unique()) == {""}:
        gene_weight = frame.groupby("gene")[column].max()
        for gene, value in gene_weight.items():
            if gene in weights.index and np.isfinite(value):
                weights.loc[gene, :] = float(value)
    else:
        for _, row in frame.iterrows():
            gene = str(row["gene"])
            ct = str(row["cell_type"])
            value = row[column]
            if gene in weights.index and ct in weights.columns and np.isfinite(value):
                weights.loc[gene, ct] = float(value)

    arr = weights.to_numpy(dtype=np.float32, copy=True)
    arr = np.maximum(arr, float(min_weight)).astype(np.float32)
    if normalize:
        mean = float(arr.mean())
        if mean > EPS:
            arr = (arr / mean).astype(np.float32)
    log(
        f"Loaded gene-CT weights {weight_file.name}: "
        f"min={arr.min():.4g} median={np.median(arr):.4g} mean={arr.mean():.4g} max={arr.max():.4g}"
    )
    return torch.from_numpy(arr).to(device)


def _normalize_mode_list(modes: str) -> list[str]:
    return [part.strip() for part in str(modes).split(",") if part.strip()]


def _composition_null_fraction(
    frac: torch.Tensor,
    mode: str,
    cts: list[str],
    shift_factor: float,
) -> torch.Tensor:
    mode = str(mode)
    if mode in {"independent", "null", "permute"}:
        if frac.shape[0] <= 1:
            return frac
        return frac.index_select(0, torch.randperm(frac.shape[0], device=frac.device))
    if "Inhibitory" not in cts:
        raise ValueError("composition null augmentation requires an Inhibitory cell type for low/high modes.")
    inh_idx = cts.index("Inhibitory")
    out = frac.clone()
    factor = max(float(shift_factor), 1e-3)
    if mode in {"low_inh", "lowInh"}:
        out[:, inh_idx] = out[:, inh_idx] * factor
    elif mode in {"high_inh", "highInh"}:
        out[:, inh_idx] = out[:, inh_idx] / factor
    else:
        raise ValueError(f"Unknown composition null augmentation mode: {mode}")
    return out / out.sum(dim=1, keepdim=True).clamp(min=1e-6)


def sample_paired_remix_fractions(
    frac: torch.Tensor,
    mode: str,
    strength: float,
    min_clip: float = 1e-6,
) -> torch.Tensor:
    """Draw realistic counterfactual fractions independently of CTS expression."""
    if frac.ndim != 2:
        raise ValueError(f"Expected a donor-by-cell-type fraction matrix, got shape={tuple(frac.shape)}")
    if frac.shape[0] <= 1:
        return frac
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(f"paired remix strength must be in [0, 1], got {strength}")

    mode = str(mode)
    if mode == "permute":
        target = frac.index_select(0, torch.randperm(frac.shape[0], device=frac.device))
    elif mode == "independent":
        # Independently permuting each axis broadens the empirical composition
        # support while retaining realistic marginal fraction distributions.
        target = torch.stack(
            [frac[:, ci].index_select(0, torch.randperm(frac.shape[0], device=frac.device)) for ci in range(frac.shape[1])],
            dim=1,
        )
    else:
        raise ValueError(f"Unknown paired remix mode: {mode}")

    out = (1.0 - float(strength)) * frac + float(strength) * target
    out = torch.clamp(out, min=float(min_clip))
    return out / out.sum(dim=1, keepdim=True).clamp(min=float(min_clip))


def make_paired_remix_augmented_features(
    bundle: dict[str, object],
    donor_feat: torch.Tensor,
    local_feat: torch.Tensor,
    y_abs: torch.Tensor,
    row_donor_idx: torch.Tensor,
    row_gene_idx: torch.Tensor,
    mode: str,
    strength: float,
    min_clip: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Recompose bulk under remixed fractions while holding the CTS truth fixed."""
    donor_ids, inverse = torch.unique(row_donor_idx, sorted=True, return_inverse=True)
    frac_base_unique = bundle["frac_matrix"].index_select(0, donor_ids)
    frac_aug_unique = sample_paired_remix_fractions(frac_base_unique, mode, strength, min_clip)
    frac_aug = frac_aug_unique.index_select(0, inverse)
    frac_log_aug = transform_fraction_torch(
        frac_aug,
        str(bundle.get("fraction_transform_mode", FRACTION_TRANSFORM_LEGACY_LOG)),
        float(bundle.get("fraction_transform_tau", 0.01)),
        float(bundle.get("fraction_min_clip", min_clip)),
    )

    expression_transform = str(bundle["expression_transform"])
    true_lin = inverse_transform_expression_torch(y_abs, expression_transform)
    bulk_star_log = transform_expression_torch(torch.sum(frac_aug * true_lin, dim=1), expression_transform)

    ref_lin_rows = bundle["ref_lin_matrix"].index_select(0, row_gene_idx)
    expected_star_lin = torch.sum(frac_aug * ref_lin_rows, dim=1)
    expected_star_log = transform_expression_torch(expected_star_lin, expression_transform)
    resid_star_log = bulk_star_log - expected_star_log

    bulk_mu = bundle["bulk_mu"].index_select(0, row_gene_idx)
    bulk_sd = bundle["bulk_sd"].index_select(0, row_gene_idx)
    bulk_scaled = (bulk_star_log - bulk_mu) / bulk_sd
    expected_scaled = (expected_star_log - bundle["expected_mu"]) / bundle["expected_sd"]
    resid_scaled = (resid_star_log - bundle["resid_mu"]) / bundle["resid_sd"]

    donor_aug = donor_feat.clone()
    donor_aug[:, 0] = bulk_scaled
    donor_aug[:, 1] = expected_scaled
    donor_aug[:, 2] = resid_scaled
    n_ct = frac_aug.shape[1]
    donor_aug[:, 3 : 3 + n_ct] = frac_aug
    donor_aug[:, 3 + n_ct : 3 + 2 * n_ct] = frac_log_aug

    local_aug = local_feat.clone()
    local_aug[:, :, 0] = frac_aug
    local_aug[:, :, 1] = frac_log_aug

    # Rebuild the donor-level summary from the complete CTS profile, not just
    # the genes present in this minibatch. This matters for fraction-refiner
    # variants; canonical V0 ignores this input but should remain physically
    # consistent under the same augmentation.
    n_donors, n_genes, n_ct = bundle["shapes"]
    truth_all = bundle["Y_abs"].reshape(n_donors, n_genes, n_ct).index_select(0, donor_ids)
    truth_all_lin = inverse_transform_expression_torch(truth_all, expression_transform)
    bulk_star_all_lin = torch.sum(frac_aug_unique[:, None, :] * truth_all_lin, dim=2)
    bulk_star_all_log = transform_expression_torch(bulk_star_all_lin, expression_transform)
    expected_star_all_lin = frac_aug_unique @ bundle["ref_lin_matrix"].transpose(0, 1)
    expected_star_all_log = transform_expression_torch(expected_star_all_lin, expression_transform)
    resid_star_all_log = bulk_star_all_log - expected_star_all_log
    donor_summary_raw = torch.stack(
        [
            bulk_star_all_log.mean(dim=1),
            bulk_star_all_log.std(dim=1, unbiased=False),
            expected_star_all_log.mean(dim=1),
            expected_star_all_log.std(dim=1, unbiased=False),
            resid_star_all_log.mean(dim=1),
            resid_star_all_log.std(dim=1, unbiased=False),
        ],
        dim=1,
    )
    donor_summary_aug_unique = (
        donor_summary_raw - bundle["donor_summary_mu"]
    ) / bundle["donor_summary_sd"]
    donor_summary_aug = donor_summary_aug_unique.index_select(0, inverse)
    stats = {
        "frac_base": frac_base_unique,
        "frac_aug": frac_aug_unique,
        "frac_mean_abs_change": torch.mean(torch.abs(frac_aug_unique - frac_base_unique)),
        "bulk_star_log": bulk_star_log,
        "expected_star_log": expected_star_log,
    }
    return donor_aug, local_aug, donor_summary_aug, stats


def compute_paired_remix_augmentation_loss(
    model: nn.Module,
    bundle: dict[str, object],
    donor_feat: torch.Tensor,
    gene_feat: torch.Tensor,
    local_feat: torch.Tensor,
    donor_summary_feat: torch.Tensor | None,
    y_abs: torch.Tensor,
    ref_mean: torch.Tensor,
    pred_abs: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    lambda_contrast: float,
    residual_loss_weight: float,
    huber_delta: float,
    row_donor_idx: torch.Tensor,
    row_gene_idx: torch.Tensor,
    loss_reduction: str,
    mode: str,
    strength: float,
    supervised_weight: float = 1.0,
    consistency_weight: float = 1.0,
    gene_ct_weight: torch.Tensor | None = None,
    residual_prior_scale: torch.Tensor | None = None,
    residual_prior_mode: str = "none",
    residual_prior_multiplier: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train on a physically valid remix and keep both predictions aligned."""
    donor_aug, local_aug, donor_summary_aug, remix_stats = make_paired_remix_augmented_features(
        bundle,
        donor_feat,
        local_feat,
        y_abs,
        row_donor_idx,
        row_gene_idx,
        mode,
        strength,
    )
    delta_aug, pred_aug, _ = forward_pred(
        model,
        donor_aug,
        gene_feat,
        local_aug,
        ref_mean,
        donor_summary_aug,
        residual_prior_scale=residual_prior_scale,
        residual_prior_mode=residual_prior_mode,
        residual_prior_multiplier=residual_prior_multiplier,
    )
    supervised_loss, _ = compute_weighted_losses(
        delta_aug,
        pred_aug,
        y_abs,
        ref_mean,
        pair_i,
        pair_j,
        lambda_contrast,
        residual_loss_weight,
        huber_delta,
        gene_ct_weight,
        row_donor_idx=row_donor_idx,
        loss_reduction=loss_reduction,
    )
    consistency_loss = F.smooth_l1_loss(pred_aug, pred_abs.detach(), beta=huber_delta)
    total = float(supervised_weight) * supervised_loss + float(consistency_weight) * consistency_loss
    stats = {
        "paired_remix_supervised_loss": float(supervised_loss.detach().cpu()),
        "paired_remix_consistency_loss": float(consistency_loss.detach().cpu()),
        "paired_remix_mean_abs_prediction_change": float(torch.mean(torch.abs(pred_aug - pred_abs)).detach().cpu()),
        "paired_remix_mean_abs_fraction_change": float(remix_stats["frac_mean_abs_change"].detach().cpu()),
    }
    return total, stats


def make_composition_null_augmented_features(
    bundle: dict[str, object],
    donor_feat: torch.Tensor,
    local_feat: torch.Tensor,
    y_abs: torch.Tensor,
    ref_mean: torch.Tensor,
    mode: str,
    shift_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cts = [str(x) for x in bundle["cts"]]
    n_ct = len(cts)
    frac_aug = _composition_null_fraction(donor_feat[:, 3 : 3 + n_ct], mode, cts, shift_factor)
    frac_log_aug = torch.log(torch.clamp(frac_aug, min=1e-6))
    expression_transform = str(bundle["expression_transform"])
    true_lin = inverse_transform_expression_torch(y_abs, expression_transform)
    ref_lin = inverse_transform_expression_torch(ref_mean, expression_transform)
    bulk_star_log = transform_expression_torch(torch.sum(frac_aug * true_lin, dim=1), expression_transform)
    expected_star_log = transform_expression_torch(torch.sum(frac_aug * ref_lin, dim=1), expression_transform)
    resid_scaled = ((bulk_star_log - expected_star_log) - bundle["resid_mu"]) / bundle["resid_sd"]

    donor_aug = donor_feat.clone()
    donor_aug[:, :3] = 0.0
    donor_aug[:, 2] = resid_scaled
    donor_aug[:, 3 : 3 + n_ct] = frac_aug
    donor_aug[:, 3 + n_ct : 3 + 2 * n_ct] = frac_log_aug

    local_aug = local_feat.clone()
    local_aug[:, :, 0] = frac_aug
    local_aug[:, :, 1] = frac_log_aug
    return donor_aug, local_aug


def compute_composition_null_augmentation_loss(
    model: nn.Module,
    bundle: dict[str, object],
    donor_feat: torch.Tensor,
    gene_feat: torch.Tensor,
    local_feat: torch.Tensor,
    donor_summary_feat: torch.Tensor | None,
    y_abs: torch.Tensor,
    ref_mean: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    lambda_contrast: float,
    residual_loss_weight: float,
    huber_delta: float,
    row_donor_idx: torch.Tensor | None,
    loss_reduction: str,
    modes: str,
    shift_factor: float,
    residual_prior_scale: torch.Tensor | None = None,
    residual_prior_mode: str = "none",
    residual_prior_multiplier: float = 1.0,
) -> torch.Tensor:
    mode_list = _normalize_mode_list(modes)
    if not mode_list:
        return ref_mean.new_tensor(0.0)
    losses = []
    for mode in mode_list:
        donor_aug, local_aug = make_composition_null_augmented_features(
            bundle,
            donor_feat,
            local_feat,
            y_abs,
            ref_mean,
            mode,
            shift_factor,
        )
        delta_aug, pred_aug, _ = forward_pred(
            model,
            donor_aug,
            gene_feat,
            local_aug,
            ref_mean,
            donor_summary_feat,
            residual_prior_scale=residual_prior_scale,
            residual_prior_mode=residual_prior_mode,
            residual_prior_multiplier=residual_prior_multiplier,
        )
        loss_aug, _ = compute_losses(
            delta_aug,
            pred_aug,
            y_abs,
            ref_mean,
            pair_i,
            pair_j,
            lambda_contrast,
            residual_loss_weight,
            huber_delta,
            row_donor_idx=row_donor_idx,
            loss_reduction=loss_reduction,
        )
        losses.append(loss_aug)
    return torch.stack(losses).mean()


def predict_absolute_tensor(
    model: nn.Module,
    bundle: dict[str, object],
    batch_size: int,
    residual_prior_mode: str = "none",
    residual_prior_multiplier: float = 1.0,
) -> torch.Tensor:
    model.eval()
    preds = []
    with torch.no_grad():
        donor_feat = bundle["donor_feat"]
        gene_feat = bundle["gene_feat"]
        local_feat = bundle["local_feat"]
        donor_summary_feat = bundle.get("donor_summary_feat")
        ref_mean = bundle["ref_mean"]
        row_gene_idx = bundle["row_gene_idx"]
        for start in range(0, donor_feat.shape[0], batch_size):
            stop = min(start + batch_size, donor_feat.shape[0])
            prior_scale = None
            if residual_prior_mode != "none":
                prior_scale = select_residual_prior_tensor(bundle, row_gene_idx[start:stop], residual_prior_mode)
            _, pred_abs, _ = forward_pred(
                model,
                donor_feat[start:stop],
                gene_feat[start:stop],
                local_feat[start:stop],
                ref_mean[start:stop],
                None if donor_summary_feat is None else donor_summary_feat[start:stop],
                residual_prior_scale=prior_scale,
                residual_prior_mode=residual_prior_mode,
                residual_prior_multiplier=residual_prior_multiplier,
            )
            preds.append(pred_abs.detach().cpu())
    return torch.cat(preds, dim=0)


def flatten_scores_to_frame(pred_arr: np.ndarray, donors: list[str], genes: list[str], cts: list[str]) -> pd.DataFrame:
    flat = pred_arr.reshape(len(donors), len(genes) * len(cts))
    cols = [f"{g}_{ct}" for g in genes for ct in cts]
    return pd.DataFrame(flat, index=pd.Index(donors, name="donor"), columns=cols)


def predict_to_frame(
    model: nn.Module,
    bundle: dict[str, object],
    batch_size: int,
    residual_prior_mode: str = "none",
    residual_prior_multiplier: float = 1.0,
) -> pd.DataFrame:
    pred_flat = predict_absolute_tensor(
        model,
        bundle,
        batch_size,
        residual_prior_mode=residual_prior_mode,
        residual_prior_multiplier=residual_prior_multiplier,
    ).numpy()
    pred_arr = pred_flat.reshape(bundle["shapes"][0], bundle["shapes"][1], bundle["shapes"][2])
    return flatten_scores_to_frame(pred_arr, bundle["donors"], bundle["genes"], bundle["cts"])


def corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    denom = float(np.sqrt((x_dev * x_dev).sum() * (y_dev * y_dev).sum()))
    if denom <= 0:
        return float("nan")
    return float((x_dev * y_dev).sum() / denom)


def corrcoef_weighted_safe(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    weight = weight[mask]
    weight_sum = float(weight.sum())
    if weight_sum <= 0:
        return float("nan")
    x_dev = x - float(np.sum(weight * x) / weight_sum)
    y_dev = y - float(np.sum(weight * y) / weight_sum)
    denom = float(np.sqrt(np.sum(weight * x_dev * x_dev) * np.sum(weight * y_dev * y_dev)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(weight * x_dev * y_dev) / denom)


def summarize_all_pcc(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    genes: list[str],
    cts: list[str],
    expression_transform: str = EXPRESSION_TRANSFORM_LOG2P1_NONNEG,
    observation_weights: pd.DataFrame | None = None,
) -> tuple[list[dict[str, float]], float]:
    pred = pred_df.copy()
    truth = truth_df.copy()
    pred.index = pred.index.astype(str)
    truth.index = truth.index.astype(str)
    pred.columns = pred.columns.astype(str)
    truth.columns = truth.columns.astype(str)
    donors = pred.index.intersection(truth.index)
    pred = pred.loc[donors]
    truth = truth.loc[donors]
    aligned_weights = None
    if observation_weights is not None:
        aligned_weights = observation_weights.copy()
        aligned_weights.index = aligned_weights.index.astype(str)
        aligned_weights.columns = aligned_weights.columns.astype(str)
        missing_donors = set(donors).difference(aligned_weights.index)
        missing_cts = set(cts).difference(aligned_weights.columns)
        if missing_donors or missing_cts:
            raise ValueError(
                "observation_weights do not cover PCC inputs: "
                f"missing_donors={len(missing_donors)} missing_cts={sorted(missing_cts)}"
            )
        aligned_weights = aligned_weights.loc[donors, cts]

    rows: list[dict[str, float]] = []
    for ct in cts:
        cols = [f"{g}_{ct}" for g in genes if f"{g}_{ct}" in pred.columns and f"{g}_{ct}" in truth.columns]
        if not cols:
            rows.append({"cell_type": ct, "median_pcc": float("nan"), "n_genes": 0})
            continue
        truth_log = transform_expression_numpy(
            truth.loc[:, cols].to_numpy(dtype=np.float32, copy=False),
            expression_transform,
        )
        pred_vals = pred.loc[:, cols].to_numpy(dtype=float, copy=False)
        pccs = []
        ct_weight = None if aligned_weights is None else aligned_weights.loc[:, ct].to_numpy(dtype=float)
        for idx in range(len(cols)):
            pcc = (
                corrcoef_safe(truth_log[:, idx], pred_vals[:, idx])
                if ct_weight is None
                else corrcoef_weighted_safe(truth_log[:, idx], pred_vals[:, idx], ct_weight)
            )
            if np.isfinite(pcc):
                pccs.append(pcc)
        rows.append(
            {
                "cell_type": ct,
                "median_pcc": float(np.median(pccs)) if pccs else float("nan"),
                "n_genes": int(len(pccs)),
                "n_positive_weight_donors": int(len(donors) if ct_weight is None else np.sum(ct_weight > 0)),
            }
        )
    overall = float(np.nanmean([row["median_pcc"] for row in rows]))
    return rows, overall


def summarize_fraction_coupling(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame | None,
    fractions: pd.DataFrame,
    genes: list[str],
    cts: list[str],
    expression_transform: str = EXPRESSION_TRANSFORM_LOG2P1_NONNEG,
    observation_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pred = pred_df.copy()
    if "donor" in pred.columns:
        pred = pred.set_index("donor")
    pred.index = pred.index.astype(str)
    pred.columns = pred.columns.astype(str)
    truth = None
    if truth_df is not None:
        truth = truth_df.copy()
        if "donor" in truth.columns:
            truth = truth.set_index("donor")
        truth.index = truth.index.astype(str)
        truth.columns = truth.columns.astype(str)
    frac = fractions.copy()
    frac.index = frac.index.astype(str)
    frac.columns = frac.columns.astype(str)
    donors = pred.index.intersection(frac.index)
    if truth is not None:
        donors = donors.intersection(truth.index)
    pred = pred.loc[donors]
    frac = frac.loc[donors, cts]
    if truth is not None:
        truth = truth.loc[donors]

    weights = None
    if observation_weights is not None:
        weights = observation_weights.copy()
        weights.index = weights.index.astype(str)
        weights.columns = weights.columns.astype(str)
        weights = weights.loc[donors, cts]

    rows: list[dict[str, object]] = []
    for output_ct in cts:
        output_weight = None if weights is None else weights.loc[:, output_ct].to_numpy(dtype=float)
        features = [f"{gene}_{output_ct}" for gene in genes if f"{gene}_{output_ct}" in pred.columns]
        for fraction_ct in cts:
            fraction_values = frac.loc[:, fraction_ct].to_numpy(dtype=float)
            pred_corrs: list[float] = []
            truth_corrs: list[float] = []
            for feature in features:
                pred_values = pred.loc[:, feature].to_numpy(dtype=float)
                pred_corr = (
                    corrcoef_safe(pred_values, fraction_values)
                    if output_weight is None
                    else corrcoef_weighted_safe(pred_values, fraction_values, output_weight)
                )
                if np.isfinite(pred_corr):
                    pred_corrs.append(pred_corr)
                if truth is not None and feature in truth.columns:
                    truth_values = transform_expression_numpy(
                        truth.loc[:, [feature]].to_numpy(dtype=np.float32, copy=False),
                        expression_transform,
                    )[:, 0]
                    truth_corr = (
                        corrcoef_safe(truth_values, fraction_values)
                        if output_weight is None
                        else corrcoef_weighted_safe(truth_values, fraction_values, output_weight)
                    )
                    if np.isfinite(truth_corr):
                        truth_corrs.append(truth_corr)
            median_pred = float(np.median(pred_corrs)) if pred_corrs else float("nan")
            median_truth = float(np.median(truth_corrs)) if truth_corrs else float("nan")
            rows.append(
                {
                    "output_cell_type": output_ct,
                    "fraction_cell_type": fraction_ct,
                    "is_own_fraction": output_ct == fraction_ct,
                    "median_pred_fraction_corr": median_pred,
                    "median_truth_fraction_corr": median_truth,
                    "signed_excess_corr": median_pred - median_truth,
                    "absolute_excess_corr": abs(median_pred) - abs(median_truth),
                    "n_pred_genes": len(pred_corrs),
                    "n_truth_genes": len(truth_corrs),
                    "n_positive_weight_donors": (
                        len(donors) if output_weight is None else int(np.sum(output_weight > 0))
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_gene_ct_pcc(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    genes: list[str],
    cts: list[str],
    expression_transform: str = EXPRESSION_TRANSFORM_LOG2P1_NONNEG,
    top_fraction: float = 0.80,
    observation_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pred = pred_df.copy()
    truth = truth_df.copy()
    if "donor" in pred.columns:
        pred = pred.set_index("donor")
    if "donor" in truth.columns:
        truth = truth.set_index("donor")
    pred.index = pred.index.astype(str)
    truth.index = truth.index.astype(str)
    pred.columns = pred.columns.astype(str)
    truth.columns = truth.columns.astype(str)
    donors = pred.index.intersection(truth.index)
    pred = pred.loc[donors]
    truth = truth.loc[donors]
    aligned_weights = None
    if observation_weights is not None:
        aligned_weights = observation_weights.copy()
        aligned_weights.index = aligned_weights.index.astype(str)
        aligned_weights.columns = aligned_weights.columns.astype(str)
        aligned_weights = aligned_weights.loc[donors, cts]

    rows: list[dict[str, object]] = []
    for ct in cts:
        for gene in genes:
            feature = f"{gene}_{ct}"
            if feature not in pred.columns or feature not in truth.columns:
                continue
            truth_vals = transform_expression_numpy(
                truth.loc[:, [feature]].to_numpy(dtype=np.float32, copy=False),
                expression_transform,
            )[:, 0]
            pred_vals = pred.loc[:, feature].to_numpy(dtype=float, copy=False)
            ct_weight = None if aligned_weights is None else aligned_weights.loc[:, ct].to_numpy(dtype=float)
            rows.append(
                {
                    "feature": feature,
                    "gene": gene,
                    "cell_type": ct,
                    "validation_pcc": (
                        corrcoef_safe(truth_vals, pred_vals)
                        if ct_weight is None
                        else corrcoef_weighted_safe(truth_vals, pred_vals, ct_weight)
                    ),
                    "n_val_donors": int(len(donors)),
                    "expression_transform": expression_transform,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["pct_rank_within_ct"] = np.nan
    out["selected_top80"] = False
    for _, idx in out.groupby("cell_type").groups.items():
        sub = out.loc[idx]
        finite = sub.loc[np.isfinite(sub["validation_pcc"].to_numpy(dtype=float))].copy()
        finite = finite.sort_values(["validation_pcc", "feature"], ascending=[False, True])
        if finite.empty:
            continue
        finite["pct_rank_within_ct"] = np.arange(1, len(finite) + 1, dtype=float) / float(len(finite))
        finite["selected_top80"] = finite["pct_rank_within_ct"] <= float(top_fraction)
        out.loc[finite.index, "pct_rank_within_ct"] = finite["pct_rank_within_ct"].to_numpy(dtype=float)
        out.loc[finite.index, "selected_top80"] = finite["selected_top80"].to_numpy(dtype=bool)
    out["disease_blind_confidence"] = out["selected_top80"].astype(float)
    return out.sort_values(["cell_type", "pct_rank_within_ct", "feature"], na_position="last").reset_index(drop=True)


def _frame_to_gene_ct_tensor(
    frame: pd.DataFrame,
    genes: list[str],
    cts: list[str],
    expression_transform: str | None,
) -> tuple[np.ndarray, list[str]]:
    arr_df = frame.copy()
    if "donor" in arr_df.columns:
        arr_df = arr_df.set_index("donor")
    arr_df.index = arr_df.index.astype(str)
    arr_df.columns = arr_df.columns.astype(str)
    cols = [f"{gene}_{ct}" for gene in genes for ct in cts if f"{gene}_{ct}" in arr_df.columns]
    keep_genes = [gene for gene in genes if all(f"{gene}_{ct}" in arr_df.columns for ct in cts)]
    if not keep_genes:
        raise ValueError("No complete gene-by-cell-type features available for collapse analysis.")
    cols = [f"{gene}_{ct}" for gene in keep_genes for ct in cts]
    mat = arr_df.loc[:, cols].to_numpy(dtype=np.float64, copy=False)
    if expression_transform is not None:
        mat = transform_expression_numpy(mat.astype(np.float32, copy=False), expression_transform).astype(np.float64, copy=False)
    tensor = mat.reshape(arr_df.shape[0], len(keep_genes), len(cts))
    return tensor, keep_genes


def standardized_residuals(arr: np.ndarray) -> np.ndarray:
    z = arr - arr.mean(axis=0, keepdims=True)
    sd = z.std(axis=0, ddof=1, keepdims=True)
    sd[sd < 1e-12] = np.nan
    return z / sd


def compute_ct_collapse_metrics(arr: np.ndarray, cell_types: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    arr_np = np.asarray(arr, dtype=np.float64)
    if arr_np.ndim != 3:
        raise ValueError("collapse metrics expect a rank-3 tensor.")
    if arr_np.shape[2] == len(cell_types):
        arr_gene_ct = arr_np
    elif arr_np.shape[1] == len(cell_types):
        arr_gene_ct = np.transpose(arr_np, (0, 2, 1))
    else:
        raise ValueError(f"Could not align tensor with {len(cell_types)} cell types; observed shape={arr_np.shape}.")

    z = standardized_residuals(arr_gene_ct)
    good_gene = np.isfinite(z).all(axis=(0, 2))
    z = z[:, good_gene, :]
    if z.shape[1] == 0:
        nan_metrics = {
            "n_genes_used": 0.0,
            "global_mean_offdiag_corr": float("nan"),
            "global_pc1_variance_share": float("nan"),
            "global_common_component_share": float("nan"),
            "median_gene_mean_offdiag_corr": float("nan"),
            "median_gene_pc1_variance_share": float("nan"),
            "p90_gene_pc1_variance_share": float("nan"),
        }
        return nan_metrics, pd.DataFrame(np.nan, index=cell_types, columns=cell_types)

    x = z.reshape(-1, z.shape[2])
    x = x[np.isfinite(x).all(axis=1)]
    corr = np.corrcoef(x, rowvar=False)
    off = corr[np.triu_indices(len(cell_types), 1)]
    cov = np.cov(x, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    common = np.repeat(z.mean(axis=2, keepdims=True), len(cell_types), axis=2)
    gene_pc1: list[float] = []
    gene_offdiag: list[float] = []
    for gi in range(z.shape[1]):
        xg = z[:, gi, :]
        if not np.isfinite(xg).all():
            continue
        cg = np.corrcoef(xg, rowvar=False)
        gene_offdiag.append(float(cg[np.triu_indices(len(cell_types), 1)].mean()))
        covg = np.cov(xg, rowvar=False)
        eigg = np.linalg.eigvalsh(covg)
        denom = eigg.sum()
        if denom > 1e-12:
            gene_pc1.append(float(eigg[-1] / denom))
    metrics = {
        "n_genes_used": float(z.shape[1]),
        "global_mean_offdiag_corr": float(np.nanmean(off)),
        "global_pc1_variance_share": float(eig[-1] / eig.sum()),
        "global_common_component_share": float(np.nanvar(common) / np.nanvar(z)),
        "median_gene_mean_offdiag_corr": float(np.nanmedian(gene_offdiag)),
        "median_gene_pc1_variance_share": float(np.nanmedian(gene_pc1)),
        "p90_gene_pc1_variance_share": float(np.nanpercentile(gene_pc1, 90)),
    }
    corr_df = pd.DataFrame(corr, index=cell_types, columns=cell_types)
    return metrics, corr_df


def compute_ct_collapse_summary_from_frames(
    pred_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    genes: list[str],
    cts: list[str],
    truth_expression_transform: str = EXPRESSION_TRANSFORM_LOG2P1_NONNEG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pred_df.copy()
    truth = truth_df.copy()
    if "donor" in pred.columns:
        pred = pred.set_index("donor")
    if "donor" in truth.columns:
        truth = truth.set_index("donor")
    pred.index = pred.index.astype(str)
    truth.index = truth.index.astype(str)
    donors = pred.index.intersection(truth.index)
    pred = pred.loc[donors]
    truth = truth.loc[donors]

    pred_tensor, pred_genes = _frame_to_gene_ct_tensor(pred, genes, cts, expression_transform=None)
    truth_tensor, truth_genes = _frame_to_gene_ct_tensor(
        truth,
        genes,
        cts,
        expression_transform=truth_expression_transform,
    )
    common_genes = [gene for gene in pred_genes if gene in set(truth_genes)]
    if not common_genes:
        raise ValueError("No overlapping genes between prediction and truth for collapse analysis.")
    pred_tensor, _ = _frame_to_gene_ct_tensor(pred, common_genes, cts, expression_transform=None)
    truth_tensor, _ = _frame_to_gene_ct_tensor(
        truth,
        common_genes,
        cts,
        expression_transform=truth_expression_transform,
    )
    pred_metrics, pred_corr = compute_ct_collapse_metrics(pred_tensor, cts)
    truth_metrics, truth_corr = compute_ct_collapse_metrics(truth_tensor, cts)
    summary = pd.DataFrame(
        [
            {
                "n_donors": len(donors),
                "n_genes_common": len(common_genes),
                "pred_global_mean_offdiag_corr": pred_metrics["global_mean_offdiag_corr"],
                "truth_global_mean_offdiag_corr": truth_metrics["global_mean_offdiag_corr"],
                "collapse_excess_global_mean_offdiag_corr": max(
                    0.0,
                    pred_metrics["global_mean_offdiag_corr"] - truth_metrics["global_mean_offdiag_corr"],
                )
                if np.isfinite(pred_metrics["global_mean_offdiag_corr"]) and np.isfinite(truth_metrics["global_mean_offdiag_corr"])
                else float("nan"),
                "pred_global_pc1_variance_share": pred_metrics["global_pc1_variance_share"],
                "truth_global_pc1_variance_share": truth_metrics["global_pc1_variance_share"],
                "pred_global_common_component_share": pred_metrics["global_common_component_share"],
                "truth_global_common_component_share": truth_metrics["global_common_component_share"],
                "pred_median_gene_mean_offdiag_corr": pred_metrics["median_gene_mean_offdiag_corr"],
                "truth_median_gene_mean_offdiag_corr": truth_metrics["median_gene_mean_offdiag_corr"],
                "pred_median_gene_pc1_variance_share": pred_metrics["median_gene_pc1_variance_share"],
                "truth_median_gene_pc1_variance_share": truth_metrics["median_gene_pc1_variance_share"],
            }
        ]
    )
    return summary, pred_corr, truth_corr


def evaluate_val_pcc(
    model: nn.Module,
    bundle: dict[str, object],
    batch_size: int,
    residual_prior_mode: str = "none",
    residual_prior_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    pred_df = predict_to_frame(
        model,
        bundle,
        batch_size,
        residual_prior_mode=residual_prior_mode,
        residual_prior_multiplier=residual_prior_multiplier,
    )
    truth_df = bundle["truth_raw_df"].copy()
    rows, mean_pcc = summarize_all_pcc(
        pred_df,
        truth_df,
        bundle["genes"],
        bundle["cts"],
        str(bundle.get("expression_transform", EXPRESSION_TRANSFORM_LOG2P1_NONNEG)),
        observation_weights=bundle.get("observation_weight_df"),
    )
    by_ct = pd.DataFrame(rows)
    pred_out = pred_df.reset_index()
    return by_ct, float(mean_pcc), pred_out


def pick_split_file(reference_dir: Path) -> Path:
    for name in ["split_8_2.tsv", "donor_splits.tsv"]:
        candidate = reference_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No split file found under {reference_dir}")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
