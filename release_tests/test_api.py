import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from dlunmix import DLUnmix, FitConfig, evaluate
from dlunmix.cli import main, read_matrix, write_cts
from dlunmix.synthetic import make_synthetic


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.reference, cls.target = make_synthetic()
        cls.donors = cls.reference[0].index.tolist()
        cls.config = FitConfig(candidate_epochs=(1, 2))
        cls.model = DLUnmix(cls.config).fit(*cls.reference, train_donors=cls.donors[:12], validation_donors=cls.donors[12:])

    def test_predict_without_truth_and_roundtrip(self):
        predicted = self.model.predict(*self.target[:2])
        self.assertEqual(predicted.shape, (5, 24))
        self.assertTrue(np.isfinite(predicted.to_numpy()).all())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model"
            self.model.save(path)
            restored = DLUnmix.load(path)
            np.testing.assert_array_equal(predicted, restored.predict(*self.target[:2]))
            np.testing.assert_array_equal(self.model.validation_pcc_, restored.validation_pcc_)
            text = (path / "model.json").read_text()
            self.assertNotIn(self.donors[0], text)
            with self.assertRaises(FileExistsError):
                self.model.save(path)

    def test_determinism_and_split_selection(self):
        model = DLUnmix(self.config).fit(*self.reference, train_donors=self.donors[:12], validation_donors=self.donors[12:])
        np.testing.assert_array_equal(model.predict(*self.target[:2]), self.model.predict(*self.target[:2]))
        self.assertEqual(model.selected_epochs_, max(model.history_, key=lambda x: x["mean_median_signed_pcc"])["epoch"])
        self.assertEqual(model.reference_counts_["total"], 16)

    def test_label_alignment(self):
        bulk, frac = self.target[:2]
        pred = self.model.predict(bulk, frac)
        reordered = self.model.predict(bulk.iloc[:, ::-1], frac.iloc[::-1, ::-1])
        np.testing.assert_array_equal(pred, reordered)
        with self.assertRaisesRegex(ValueError, "genes"):
            self.model.predict(bulk.iloc[:, :-1], frac)
        with self.assertRaisesRegex(ValueError, "donor"):
            self.model.predict(bulk, frac.iloc[:-1])

    def test_invalid_inputs(self):
        bulk, frac = self.target[:2]
        bad = bulk.copy(); bad.iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            self.model.predict(bad, frac)
        bad_frac = frac.copy(); bad_frac.iloc[0] *= 0.5
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.model.predict(bulk, bad_frac)
        bad = bulk.copy(); bad.index = ["duplicate"] * len(bad)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.model.predict(bad, frac)
        for floor in (0, -1, 2, float("nan")):
            with self.assertRaises(ValueError):
                self.model.predict(bulk, frac, fraction_floor=floor)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            DLUnmix(self.config).fit(*self.reference, train_donors=self.donors[:12], validation_donors=self.donors[11:])
        with self.assertRaises(ValueError):
            FitConfig(candidate_epochs=(2, 1))

    def test_selection_is_not_prediction_filter(self):
        before = self.model.predict(*self.target[:2])
        self.assertFalse(self.model.selected_profiles(1).to_numpy().any())
        np.testing.assert_array_equal(before, self.model.predict(*self.target[:2]))
        np.testing.assert_array_equal(self.model.selected_profiles(), self.model.validation_pcc_ > 0.4)

    def test_floor_changes_only_requested_predictions(self):
        bulk, frac = self.target[:2]
        frac = frac.copy(); frac.iloc[0] = [0, 0.4, 0.6]
        default = self.model.predict(bulk, frac)
        original = self.model.predict(bulk, frac, fraction_floor=1e-6)
        self.assertTrue(np.isfinite(default.to_numpy()).all())
        self.assertGreater(np.max(np.abs(default.iloc[0]-original.iloc[0])), 0)
        self.assertEqual(self.model.model_.fraction_min_clip, 1e-6)

    def test_optional_evaluation(self):
        p = self.model.predict(*self.target[:2])
        scores = evaluate(p, self.target[2])
        self.assertEqual(len(scores), 24)
        first = scores.iloc[0]
        truth = np.log2(self.target[2][first.cell_type][first.gene].to_numpy(dtype=np.float32) + 1)
        expected = np.sqrt(np.mean((truth.astype(float) - p[(first.gene, first.cell_type)].to_numpy())**2))
        self.assertAlmostEqual(first.rmse, expected)

    def test_tsv_and_cli_fit_predict(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            main(["demo", "--out", str(d / "demo")])
            demo = d / "demo"
            cts = read_matrix(demo / "reference_cts.tsv", cts=True)
            for ct in cts:
                np.testing.assert_allclose(cts[ct], self.reference[2][ct])
            main(["fit", "--reference-bulk", str(demo / "reference_bulk.tsv"), "--reference-fractions", str(demo / "reference_fractions.tsv"),
                  "--reference-cts", str(demo / "reference_cts.tsv"), "--splits", str(demo / "splits.tsv"),
                  "--candidate-epochs", "1", "2", "--out", str(d / "cli_model")])
            main(["predict", "--model", str(d / "cli_model"), "--bulk", str(demo / "target_bulk.tsv"),
                  "--fractions", str(demo / "target_fractions.tsv"), "--out", str(d / "no_truth")])
            self.assertFalse((d / "no_truth/evaluation.tsv").exists())
            self.assertEqual(json.loads((d / "no_truth/prediction.json").read_text())["fraction_floor"], 0.01)
            (d / "dup.tsv").write_text("donor\tg\tg\nx\t1\t2\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_matrix(d / "dup.tsv")
            (d / "numeric.tsv").write_text("donor\tg\n001\t2\n")
            self.assertEqual(read_matrix(d / "numeric.tsv").index.tolist(), ["001"])


if __name__ == "__main__":
    unittest.main()
