"""Opt-in software parity check against a supplied adopted source directory.

Set DLUNMIX_REFERENCE_SOURCE to the directory containing the three original
Python files. All data and generated outputs are tiny synthetic inputs in a
temporary directory. No research data, trained weights or jobs are used.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from dlunmix import DLUnmix, FitConfig
from dlunmix.api import _bundle, _truth
from dlunmix.synthetic import make_synthetic


@unittest.skipUnless(os.environ.get("DLUNMIX_REFERENCE_SOURCE"), "original reference source not provided")
class ReferenceParity(unittest.TestCase):
    def test_features_training_selection_refit_and_predictions(self):
        torch.set_num_threads(1)
        source = Path(os.environ["DLUNMIX_REFERENCE_SOURCE"])
        spec = importlib.util.spec_from_file_location("adopted_common", source / "dl_unmix_common.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        reference, target = make_synthetic()
        bulk, fractions, cts = reference
        genes, types = sorted(bulk.columns), list(fractions.columns)
        donors = bulk.index.tolist()
        truth = _truth(cts, donors, genes, types)
        target_truth = _truth(target[2], list(target[0].index), genes, types)
        model = DLUnmix(FitConfig(candidate_epochs=(1, 2))).fit(*reference, train_donors=donors[:12], validation_donors=donors[12:])
        meta = original.build_gene_meta(truth, genes, types)
        for key in meta:
            np.testing.assert_array_equal(model.gene_meta_[key], meta[key])
        old_bundle, scalers = original.build_bundle(bulk, fractions, truth, meta, None, True, torch.device("cpu"), bulk_feature_mode="bulk_resid")
        new_bundle, _ = _bundle(bulk, fractions, None, model.gene_meta_, model.scalers_)
        for key in ["donor_feat", "gene_feat", "local_feat", "ref_mean"]:
            torch.testing.assert_close(old_bundle[key], new_bundle[key], rtol=0, atol=0)
        for key in scalers:
            np.testing.assert_array_equal(scalers[key], model.scalers_[key])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "dlunmix_stage").mkdir(); (tmp / "data").mkdir()
            for frame, path in [(bulk, "dlunmix_stage/reference_meanpb_pbs.tsv"), (fractions, "dlunmix_stage/reference_meanpb_frac.tsv"),
                                (truth, "dlunmix_stage/reference_meanpb_cts.tsv"), (target[0], "data/bulk.tsv"),
                                (target[1], "data/frac.tsv"), (target_truth, "data/truth_cts.tsv")]:
                frame.to_csv(tmp / path, sep="\t", index_label="donor")
            pd.DataFrame({"donor": donors, "split": ["train"]*12 + ["val"]*4}).to_csv(tmp / "split.tsv", sep="\t", index=False)
            loaded_bulk, loaded_truth, loaded_frac = original.load_reference_stage(tmp)
            loaded_meta = original.build_gene_meta(loaded_truth, genes, types)
            loaded_bundle, _ = original.build_bundle(loaded_bulk, loaded_frac, loaded_truth, loaded_meta, None, True, torch.device("cpu"), bulk_feature_mode="bulk_resid")
            print("TSV_FEATURE_MAX_DIFFERENCES", {k: float((old_bundle[k]-loaded_bundle[k]).abs().max())
                  for k in ["donor_feat", "gene_feat", "local_feat", "Y_abs", "ref_mean"]})
            flags = ["--work-dir", str(tmp), "--bulk-feature-mode", "bulk_resid", "--fraction-feature-mode", "log_only",
                     "--variant-id", "V0_no_expected_logfrac_only"]
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
            commands = [[sys.executable, str(source / "train_step1.py"), *flags, "--split-file", str(tmp / "split.tsv"),
                         "--epochs", "2", "--save-epochs", "1,2", "--out-dir", str(tmp / "step1")]]
            for command in commands:
                r = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, r.stderr[-4000:])
            chosen = json.loads((tmp / "step1/best_epoch.json").read_text())["best_epoch"]
            self.assertEqual(chosen, model.selected_epochs_)
            r = subprocess.run([sys.executable, str(source / "train_step2.py"), *flags, "--epochs", str(chosen),
                                "--out-dir", str(tmp / "step2")], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr[-4000:])
            expected_weights = torch.load(tmp / "step2/final.pt", weights_only=True, map_location="cpu")
            self.assertEqual(set(expected_weights), set(model.model_.state_dict()))
            print("MAX_WEIGHT_DIFFERENCE", max(float((v-expected_weights[k]).abs().max()) for k, v in model.model_.state_dict().items()))
            for key, value in model.model_.state_dict().items():
                torch.testing.assert_close(value, expected_weights[key], rtol=1e-6, atol=1e-7)
            old_predictions = pd.read_csv(tmp / "step2/PerGene.tsv", sep="\t", index_col=0)
            np.testing.assert_allclose(model.predict(*target[:2], fraction_floor=1e-6), old_predictions, atol=1e-6, rtol=1e-6)
            # Direct original prediction at the adopted 1% floor, using the same
            # trained parameters and original fractions in the residual path.
            original_model = original.EarlySplitResidualNet(3+2*len(types), 2*len(types)+3, 4, len(types), 0.15,
                ct_names=types, bulk_feature_mode="bulk_resid", fraction_feature_mode="log_only", fraction_min_clip=0.01)
            original_model.load_state_dict(model.model_.state_dict())
            target_bundle, _ = original.build_bundle(*[target[0], target[1], target_truth, meta, scalers, False, torch.device("cpu")],
                                                      bulk_feature_mode="bulk_resid", fraction_min_clip=0.01)
            expected = original.predict_to_frame(original_model, target_bundle, 8192)
            np.testing.assert_array_equal(model.predict(*target[:2]), expected)


if __name__ == "__main__":
    unittest.main()
