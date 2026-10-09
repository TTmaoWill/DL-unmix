"""Optional numerical contract with the externally supplied adopted source."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dlunmix import DLUnmix, FitConfig
from dlunmix import _model
from dlunmix.api import _bundle, _truth, _network
from dlunmix.synthetic import make_synthetic


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.environ.get("DLUNMIX_REFERENCE_SOURCE"), "adopted source not provided")
class ReferenceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        source = Path(os.environ["DLUNMIX_REFERENCE_SOURCE"]) / "dl_unmix_common.py"
        provenance = json.loads((Path(__file__).resolve().parents[1] / "tests/reference_source.json").read_text())
        if hashlib.sha256(source.read_bytes()).hexdigest() != provenance["source_files_sha256"][source.name]:
            raise ValueError("adopted source hash does not match provenance")
        cls.original = load_module("adopted", source)
        cls.migration = load_module("conversion", Path(__file__).resolve().parents[1] / "tools/convert_model.py")

    def case(self, n_ct, device="cpu"):
        reference, target = make_synthetic(n_cell_types=n_ct)
        bulk, fractions, cts = reference
        genes, types = sorted(bulk.columns), list(fractions.columns)
        truth = _truth(cts, list(bulk.index), genes, types)
        old_meta = self.original.build_gene_meta(truth, genes, types)
        meta = _model.build_gene_meta(truth, genes, types)
        for new, old in [("reference_mean", "ref_mean_raw"), ("gene_features", "gene_feature_matrix"), ("local_features", "local_static")]:
            np.testing.assert_array_equal(meta[new], old_meta[old])
        old_bundle, old_scales = self.original.build_bundle(bulk, fractions, truth, old_meta, None, True, torch.device(device), bulk_feature_mode="bulk_resid")
        bundle, scales = _bundle(bulk, fractions, truth, meta, fit=True, device=device)
        for key in scales:
            np.testing.assert_array_equal(scales[key], old_scales[key])
        for key in ("gene_feat", "Y_abs", "ref_mean"):
            torch.testing.assert_close(bundle[key], old_bundle[key], atol=0, rtol=0)
        torch.testing.assert_close(bundle["donor_feat"], old_bundle["donor_feat"][:, [0,2]+list(range(3,3+n_ct))], atol=0, rtol=0)
        torch.testing.assert_close(bundle["local_feat"], old_bundle["local_feat"][:,:,2:], atol=0, rtol=0)
        cfg = FitConfig(candidate_epochs=(1,2), batch_size=37)
        _model.set_determinism(cfg.seed)
        old = self.original.EarlySplitResidualNet(3+2*n_ct, 2*n_ct+3, 4, n_ct, cfg.dropout,
            ct_names=types, bulk_feature_mode="bulk_resid", fraction_feature_mode="log_only").to(device)
        new = _network(n_ct, cfg, device=device)
        new.load_state_dict(self.migration.convert_weights(old.state_dict(), n_ct))
        return reference, target, truth, old_meta, meta, old_scales, scales, old_bundle, bundle, cfg, old, new

    def test_mapped_inference_and_training(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for c in (2,3,5):
                with self.subTest(device=device, n_ct=c):
                    reference,target,truth,om,meta,oscale,scales,ob,b,cfg,old,new = self.case(c,device)
                    for floor in (1e-6,0.01):
                        old.fraction_min_clip = new.fraction_floor = floor
                        for zeros in (False,True):
                            fractions = target[1].copy()
                            if zeros:
                                fractions.iloc[0,:] = 0
                                fractions.iloc[0,-1] = 1
                            target_truth = _truth(target[2],list(target[0].index),list(meta["genes"]),list(meta["cts"]))
                            ot,_ = self.original.build_bundle(target[0],fractions,target_truth,om,oscale,False,torch.device(device),bulk_feature_mode="bulk_resid",fraction_min_clip=floor)
                            nt,_ = _bundle(target[0],fractions,None,meta,scales,device=device)
                            expected = self.original.predict_to_frame(old,ot,17)
                            actual = _model.predict_to_frame(new,nt,17)
                            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=2e-5)
                            print("MAPPED_INFERENCE",device,c,floor,zeros,float(np.abs(actual.to_numpy()-expected.to_numpy()).max()))
                    old.fraction_min_clip = new.fraction_floor = 1e-6
                    optimizer_old = torch.optim.AdamW(old.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
                    optimizer_new = torch.optim.AdamW(new.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
                    # One common row order; replay the dropout RNG state for each paired update.
                    indices = torch.randperm(len(b["donor_feat"]),generator=torch.Generator().manual_seed(cfg.seed)).to(device)
                    pairs = tuple(x.to(device) for x in _model.build_pair_indices(c))
                    old.train(); new.train()
                    max_loss = 0.0
                    for epoch in range(3):
                        for idx in indices.split(cfg.batch_size):
                            cpu_state = torch.get_rng_state()
                            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                            optimizer_old.zero_grad(); optimizer_new.zero_grad()
                            delta,pred,_ = self.original.forward_pred(old,ob["donor_feat"][idx],ob["gene_feat"][idx],ob["local_feat"][idx],ob["ref_mean"][idx])
                            loss_old,_ = self.original.compute_losses(delta,pred,ob["Y_abs"][idx],ob["ref_mean"][idx],*pairs,1.,1.,1.)
                            loss_old.backward(); optimizer_old.step()
                            torch.set_rng_state(cpu_state)
                            if cuda_state is not None:
                                torch.cuda.set_rng_state_all(cuda_state)
                            delta_new = new(b["donor_feat"][idx],b["gene_feat"][idx],b["local_feat"][idx])
                            loss_new = _model.training_loss(delta_new,b["Y_abs"][idx],b["ref_mean"][idx],*pairs)
                            loss_new.backward(); optimizer_new.step()
                            max_loss=max(max_loss,abs(float(loss_old.detach())-float(loss_new.detach())))
                            torch.testing.assert_close(loss_new,loss_old,rtol=1e-5,atol=2e-5)
                    mapped = self.migration.convert_weights(old.state_dict(),c)
                    diff=max(float((v.detach().cpu()-mapped[k]).abs().max()) for k,v in new.state_dict().items())
                    for k,v in new.state_dict().items():
                        torch.testing.assert_close(v.detach().cpu(),mapped[k],rtol=1e-5,atol=2e-5)
                    expected = self.original.predict_to_frame(old,ob,8192)
                    actual = _model.predict_to_frame(new,b,8192)
                    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=2e-5)
                    old_scores,old_score,_ = self.original.evaluate_val_pcc(old,ob,8192)
                    scores,score,_ = _model.evaluate_val_pcc(new,b,8192)
                    self.assertAlmostEqual(score,old_score,places=5)
                    print("CONTROLLED_TRAINING",device,c,"loss",max_loss,"weights",diff,"predictions",float(np.abs(actual.to_numpy()-expected.to_numpy()).max()),"score",abs(score-old_score))

    def test_artifact_conversion(self):
        reference,target,truth,om,meta,oscale,scales,ob,b,cfg,old,new = self.case(3)
        donors=list(reference[0].index)
        template=DLUnmix(cfg).fit(*reference,train_donors=donors[:12],validation_donors=donors[12:])
        with tempfile.TemporaryDirectory() as tmp:
            source,destination=Path(tmp)/"source",Path(tmp)/"converted"
            template.save(source)
            data=json.loads((source/"model.json").read_text())
            data.update(format_version=1,package_version="0.2.0",variant="V0_no_expected_logfrac_only")
            data.pop("architecture")
            (source/"model.json").write_text(json.dumps(data))
            arrays={"meta__"+k:v for k,v in om.items() if k not in ("genes","cts")}
            arrays.update({"scaler__"+k:np.asarray(v) for k,v in oscale.items()})
            arrays["validation_pcc"]=template.validation_pcc_
            np.savez_compressed(source/"features.npz",**arrays)
            torch.save(old.state_dict(),source/"weights.pt")
            with self.assertRaisesRegex(ValueError,"unsupported model artifact"):
                DLUnmix.load(source)
            converted=self.migration.convert(source,destination)
            expected=_model.predict_to_frame(new,b,8192)
            actual=converted.predict(*reference[:2],fraction_floor=1e-6)
            np.testing.assert_array_equal(actual.to_numpy(),expected.to_numpy())
            restored=DLUnmix.load(destination)
            np.testing.assert_array_equal(actual,restored.predict(*reference[:2],fraction_floor=1e-6))
            np.testing.assert_array_equal(converted.validation_pcc_,template.validation_pcc_)
            with self.assertRaises(FileExistsError):
                self.migration.convert(source,destination)


if __name__ == "__main__":
    unittest.main()
