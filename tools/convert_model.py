"""Convert a trusted DL-unmix 0.2 artifact to the current model format.

Run from a checkout with DL-unmix installed:
    python tools/convert_model.py source-model destination-model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from dlunmix import DLUnmix, FitConfig
from dlunmix.api import _network, _labels


def convert_weights(weights, n_ct):
    """Select the learned inputs in a format-1 state dictionary."""
    shared_columns = [0, 2] + list(range(3+n_ct, 6+4*n_ct))
    head_columns = list(range(48)) + [49, 50, 51]
    if weights["shared.0.weight"].shape != (64, 6+4*n_ct):
        raise ValueError("unexpected source shared-layer shape")
    converted = {k: v.detach().cpu().clone() for k, v in weights.items()}
    converted["shared.0.weight"] = converted["shared.0.weight"][:, shared_columns].contiguous()
    for i in range(n_ct):
        key = f"ct_towers.{i}.0.weight"
        if converted[key].shape != (64, 52):
            raise ValueError("unexpected source head shape")
        converted[key] = converted[key][:, head_columns].contiguous()
    return converted


def convert(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    data = json.loads((source / "model.json").read_text())
    if data.get("format_version") != 1 or data.get("variant") != "V0_no_expected_logfrac_only":
        raise ValueError("source must be an adopted format-1 DL-unmix model")
    if data.get("expression_transform") != "log2p1_nonnegative" or data.get("training_fraction_floor") != 1e-6:
        raise ValueError("unsupported source preprocessing")
    config = dict(data["config"])
    config["candidate_epochs"] = tuple(config["candidate_epochs"])
    obj = DLUnmix(FitConfig(**config))
    obj.genes_ = _labels(data["genes"], "model genes")
    obj.cell_types_ = _labels(data["cell_types"], "model cell types")
    with np.load(source / "features.npz", allow_pickle=False) as arrays:
        obj.gene_meta_ = {
            "reference_mean": arrays["meta__ref_mean_raw"].copy(),
            "gene_features": arrays["meta__gene_feature_matrix"].copy(),
            "local_features": arrays["meta__local_static"].copy(),
            "genes": np.asarray(obj.genes_, dtype=object),
            "cts": np.asarray(obj.cell_types_, dtype=object),
        }
        obj.scalers_ = {k: arrays["scaler__"+k].copy() for k in ("bulk_mu", "bulk_sd", "resid_mu", "resid_sd")}
        obj.validation_pcc_ = arrays["validation_pcc"].copy()
    obj.model_ = _network(len(obj.cell_types_), obj.config)
    weights = torch.load(source / "weights.pt", map_location="cpu", weights_only=True)
    obj.model_.load_state_dict(convert_weights(weights, len(obj.cell_types_)), strict=True)
    obj.model_.eval()
    obj.selected_epochs_ = data["selected_epochs"]
    obj.history_ = data["history"]
    obj.reference_counts_ = data["reference_counts"]
    # Validate the complete artifact before writing the requested destination.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "model"
        obj.save(staged)
        checked = DLUnmix.load(staged)
    checked.save(destination)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    convert(args.source, args.destination)
    print(f"Converted model saved to {args.destination}")


if __name__ == "__main__":
    main()
