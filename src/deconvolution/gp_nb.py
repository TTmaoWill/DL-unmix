#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DL‑unmix – negative‑binomial multi‑task Gaussian‑process deconvolution
=====================================================================
• Replaces the ExactGP + Gaussian likelihood with a Variational GP + NB likelihood.
• Keeps batching, gene permutation, and I/O interface identical to the original script.
---------------------------------------------------------------------
"""
import os, random, math, warnings, argparse
import torch, gpytorch
import numpy  as np
import pandas as pd

# ------------------------------------------------------------------ #
#                PRE‑PROCESSING  (unchanged vs original)             #
# ------------------------------------------------------------------ #
from src.preprocess.generate_pseudobulk import create_pb           # your helper

# ---------------------------  NB likelihood ----------------------- #
class NegativeBinomialLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    r"""
    NB( y | μ, r )    with log‑link   μ = exp(f)
    If you want gene‑specific dispersions, make `log_r` a vector of size (nt,).
    """
    def __init__(self, init_r: float = 10.0):
        super().__init__()
        self.register_parameter(
            "log_r", torch.nn.Parameter(torch.tensor(math.log(init_r)))
        )

    def forward(self, function_samples, **kwargs):
        rate = function_samples.exp()                # μ = exp(f)
        r     = self.log_r.exp()                     # dispersion > 0
        p     = r / (r + rate)                      # NB (total_count=r, probs=p)
        return torch.distributions.NegativeBinomial(total_count=r, probs=p)

# ------------------------  Variational MT‑GP ---------------------- #
from gpytorch.variational import (
    MeanFieldVariationalDistribution,
    VariationalStrategy,
    IndependentMultitaskVariationalStrategy,
)
class MultitaskNBGP(gpytorch.models.ApproximateGP):
    """
    Sparse variational GP with a separate latent function per task (gene × cell‑type).
    Uses an RBF kernel + independent coregionalisation (rank‑1).  For faster
    experiments, feel free to replace the kernel with an `gpytorch.kernels.LinearKernel`.
    """
    def __init__(self, inducing_points: torch.Tensor, num_tasks: int):
        # set up q(u)
        q_dist = MeanFieldVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
        )
        base_vs = VariationalStrategy(
            self, inducing_points, q_dist, learn_inducing_locations=True
        )
        vs = IndependentMultitaskVariationalStrategy(base_vs, num_tasks=num_tasks)
        super().__init__(vs)

        self.mean_module  = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.keops.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

# ------------------------------------------------------------------ #
#                        MAIN PIPELINE                               #
# ------------------------------------------------------------------ #
def run_DLunmix_NBGP(target_dir: str,
                     sc_dir: str,
                     out_dir: str,
                     batch_size: int = 500,
                     log_normalization: bool = True,
                     seed: int = -1,
                     iters: int = 1500,
                     lr: float = 1e-2,
                     n_inducing: int = 800):
    """Negative‑binomial GP version of DL‑unmix."""
    print("▶  DL‑unmix NB‑GP estimation started")
    target_bulk = pd.read_csv(f"{target_dir}_pbs.tsv",  sep="\t", index_col=0)
    target_frac = pd.read_csv(f"{target_dir}_frac.tsv", sep="\t", index_col=0)
    cell_types  = np.sort(target_frac.columns.values)

    sc_count = pd.read_csv(f"{sc_dir}_count.tsv",     sep="\t", index_col=0).T
    sc_meta  = pd.read_csv(f"{sc_dir}_metadata.tsv",  sep="\t")
    sc_meta.columns = ['cell_name', 'cell_type', 'donor']
    prefix   = os.path.basename(sc_dir)
    sc_pbs, sc_cts, sc_frac = create_pb(
        sc_count, sc_meta, prefix=prefix,
        out_dir=os.path.dirname(sc_dir), qc_threshold=0.8
    )

    # ---------- gene intersection & orientation ----------
    common_genes = np.intersect1d(target_bulk.index, sc_pbs.index)
    assert len(common_genes) > 0, "No overlapping genes."
    target_bulk = target_bulk.loc[common_genes].T
    sc_pbs      = sc_pbs.loc[common_genes].T
    sc_cts      = sc_cts.loc[[f"{g}_{ct}" for g in common_genes for ct in cell_types]].T
    print(f"✓  {len(common_genes)} common genes retained")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("✓  Using device:", device)

    # ---------- optional gene permutation for robustness ----------
    original_genes  = sc_pbs.columns.values
    permuted_genes  = original_genes.copy()
    if seed > 0:
        random.seed(seed)
        random.shuffle(permuted_genes)
        target_bulk  = target_bulk[permuted_genes]
        sc_pbs       = sc_pbs[permuted_genes]
        sc_cts       = sc_cts[[f"{g}_{ct}" for g in permuted_genes for ct in cell_types]]

    # ---------- log‑normalisation (optional) ----------
    if log_normalization:
        target_bulk = np.log1p(target_bulk)
        sc_cts      = np.log1p(sc_cts)
        sc_pbs      = np.log1p(sc_pbs)

    pred = pd.DataFrame(0, index=target_bulk.index,
                           columns=sc_cts.columns, dtype=np.float32)

    n_genes, n_samples = target_bulk.shape[1], target_bulk.shape[0]
    n_batches          = int(np.ceil(n_genes / batch_size))
    print(f"✓  Processing {n_genes} genes in {n_batches} batch(es)")

    # ------------------------------------------------------------------ #
    #                        BATCHED TRAINING                            #
    # ------------------------------------------------------------------ #
    for b in range(n_batches):
        lower, upper = b * batch_size, min((b + 1) * batch_size, n_genes)
        genes = permuted_genes[lower:upper]
        print(f"── Batch {b + 1}/{n_batches} [{lower}:{upper}]  ({len(genes)} genes)")

        # -------- prepare design matrices --------
        X_train = np.hstack([sc_pbs[genes].to_numpy(), sc_frac.to_numpy()]).astype(np.float32)
        Y_train = sc_cts[[f"{g}_{ct}" for g in genes for ct in cell_types]].to_numpy().astype(np.float32)

        X_test  = np.hstack([target_bulk[genes].to_numpy(), target_frac.to_numpy()]).astype(np.float32)

        X_train = torch.from_numpy(X_train).to(device)
        Y_train = torch.from_numpy(Y_train).to(device)
        X_test  = torch.from_numpy(X_test).to(device)

        nt = Y_train.shape[1]                     # tasks = genes × cell types

        # -------- model & likelihood --------
        inducing = X_train[:n_inducing].clone()
        likelihood = NegativeBinomialLikelihood(init_r=10.0).to(device)
        model      = MultitaskNBGP(inducing, nt).to(device)

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train.size(0))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # -------- training loop --------
        model.train(); likelihood.train()
        for i in range(1, iters + 1):
            optimizer.zero_grad()
            out   = model(X_train)
            loss  = -mll(out, Y_train)
            loss.backward()
            optimizer.step()
            if i % 100 == 0 or i == 1:
                print(f"   iter {i:4d}/{iters}  |  ELBO: {-loss.item():.3f}")

        # -------- prediction --------
        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_pred = likelihood(model(X_test)).mean.cpu().numpy()
            y_pred = np.maximum(y_pred, 0.0)

        pred_cols = [f"{g}_{ct}" for g in genes for ct in cell_types]
        pred[pred_cols] = y_pred.astype(np.float32)

        # tidy up GPU memory
        del X_train, Y_train, X_test, model, likelihood, mll, optimizer, y_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------- restore original gene order & save --------
    pred = pred[[f"{g}_{ct}" for g in original_genes for ct in cell_types]]
    os.makedirs(out_dir, exist_ok=True)
    pred.to_csv(f"{out_dir}/nbgp_{batch_size}.tsv", sep="\t")
    print("✔  DL‑unmix NB‑GP finished")

    return pred

# ------------------------------------------------------------------ #
#                      CLI  (optional convenience)                   #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL-unmix negative-binomial GP")
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--sc_dir",     required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--lognorm",    action="store_true")
    parser.add_argument("--seed",       type=int, default=-1)
    parser.add_argument("--iters",      type=int, default=1500)
    parser.add_argument("--lr",         type=float, default=1e-2)
    parser.add_argument("--indu",       type=int, default=800)
    args = parser.parse_args()

    run_DLunmix_NBGP(
        target_dir=args.target_dir,
        sc_dir=args.sc_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        log_normalization=args.lognorm,
        seed=args.seed,
        iters=args.iters,
        lr=args.lr,
        n_inducing=args.indu,
    )
