import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os
import gpytorch  # New import for GP model
from src.preprocess.generate_pseudobulk import create_pb

# --- New GP Model definition (replaces Generator/Discriminator) ---
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nt):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=nt
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.keops.RBFKernel(), num_tasks=nt, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def run_DLunmix_GP(target_dir: str,
                    sc_dir: str,
                    out_dir: str,
                    batch_size: int,
                    log_normalization: bool = True,
                    seed: int = -1):
    """Runs the DL-unmix method using a Multitask Gaussian Process model.

    This version reuses the original framework (data preparation, permutation, etc.)
    but replaces the GAN model with a gpytorch-based Multitask GP model.

    Args:
        target_dir (str): The directory containing the target files.
        sc_dir (str): The directory containing the single-cell reference files.
        out_dir (str): The directory where the output will be saved.
        batch_size (int): The batch size for the model.
        log_normalization (bool): Whether to perform log normalization on the input data.
        seed (int): The random seed for gene permutation.
    """
    print("Starting DL-unmix GP estimation")
    target_bulk = pd.read_csv(f"{target_dir}_pbs.tsv", sep="\t", index_col=0)
    target_frac = pd.read_csv(f"{target_dir}_frac.tsv", sep="\t", index_col=0)
    cell_types = np.sort(target_frac.columns.values)

    sc_count = pd.read_csv(f"{sc_dir}_count.tsv", sep="\t", index_col=0).T
    sc_meta = pd.read_csv(f"{sc_dir}_metadata.tsv", sep="\t", index_col=0)
    sc_meta.columns = ['cell_name', 'cell_type', 'donor']
    prefix = os.path.basename(sc_dir)
    sc_pbs, sc_cts, sc_frac = create_pb(sc_count, sc_meta, prefix=prefix, out_dir=os.path.dirname(sc_dir), qc_threshold=0.8)

    # Overlap genes between target and sc data
    common_genes = np.intersect1d(target_bulk.index, sc_pbs.index)
    target_bulk = target_bulk.loc[common_genes].T
    sc_pbs = sc_pbs.loc[common_genes].T
    sc_cts = sc_cts.loc[[f"{gene}_{ct}" for gene in common_genes for ct in cell_types]].T
    print(f"Number of common genes: {len(common_genes)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, 'to run the GP model.')

    # Permutation
    original_genes = sc_pbs.columns.values
    permuted_genes = original_genes.copy() 
    if seed > 0:
        random.seed(seed)
        random.shuffle(permuted_genes)
        target_bulk = target_bulk[permuted_genes]
        sc_pbs = sc_pbs[permuted_genes]
        sc_cts = sc_cts[[f"{gene}_{ct}" for gene in permuted_genes for ct in cell_types]]

    # Initialize the output dataframe
    pred = pd.DataFrame(0, index=target_bulk.index, columns=sc_cts.columns)

    # Log normalization
    if log_normalization:
        target_bulk = np.log2(target_bulk + 1)
        sc_cts = np.log2(sc_cts + 1)
        sc_pbs = np.log2(sc_pbs + 1)

    # Get the number of genes and batches
    n_genes = target_bulk.shape[1]
    n_batch = int(np.ceil(n_genes / batch_size))
    n_pb = sc_pbs.shape[0]
    n_ct = sc_frac.shape[1]

    # Run the GP model for each gene batch
    for b in range(n_batch):
        lower = b * batch_size
        upper = np.min([(b + 1) * batch_size, n_genes])
        print("Running the " + str(b + 1) + " batch.")
        genes = permuted_genes[lower:upper]
        print("Genes:", genes)

        # Prepare training data similar to before:
        # X: [sc_pbs for genes, sc_frac]
        X = np.hstack([sc_pbs[genes].to_numpy(), sc_frac.to_numpy()])
        # Y: single-cell counts for genes across cell types (flattened)
        Y = sc_cts[[f"{gene}_{ct}" for gene in genes for ct in cell_types]].to_numpy()
        # Prepare test data from target_bulk and target_frac
        X2 = np.hstack([target_bulk[genes].to_numpy(), target_frac.to_numpy()])

        X_train = torch.FloatTensor(X).to(device)
        Y_train = torch.FloatTensor(Y).to(device)
        X_test = torch.FloatTensor(X2).to(device)

        # For the GP model, the number of tasks equals number of genes in batch times number of cell types.
        nt = Y_train.shape[1]

        # Initialize likelihood and GP model
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nt).to(device)
        model = MultitaskGPModel(X_train, Y_train, likelihood, nt).to(device)

        # Training
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(100):
            optimizer.zero_grad()
            output = model(X_train)
            loss = -mll(output, Y_train)
            loss.backward()
            print(f'Batch {b + 1}, Iter {i + 1}/100 - Loss: {loss.item():.3f}')
            optimizer.step()

        # Evaluation
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            y_pred = likelihood(model(X_test)).mean.cpu().numpy()
            y_pred = np.maximum(y_pred, 0)  # Ensure non-negative predictions

        # Store predictions: assign the predicted values to the appropriate gene-cell type columns
        pred_cols = [f"{gene}_{ct}" for gene in genes for ct in cell_types]
        pred[pred_cols] = y_pred

        # Clean up memory if needed
        del X, Y, X_train, Y_train, X_test, model, likelihood, optimizer, mll, y_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Reorder predictions to match original gene order
    pred = pred[[f"{gene}_{ct}" for gene in original_genes for ct in cell_types]]
    pred.to_csv(f"{out_dir}/gp_{batch_size}.tsv", sep="\t")
    print("DL-unmix GP estimation finished successfully.")
    return pred
