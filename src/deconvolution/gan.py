import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os
from src.preprocess.generate_pseudobulk import create_pb

# Generator model
class Generator(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[3], output_size),
            nn.ReLU()  # Ensure positive outputs
        )
    def forward(self, x):
        return self.model(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
            nn.Sigmoid()  # Output a probability (real/fake)
        )
    def forward(self, x):
        return self.model(x)

def run_DLunmix_GAN(target_dir: str,
    sc_dir: str,
    out_dir: str,
    batch_size: int,
    log_normalization: bool = True,
    seed: int = -1):
    """Runs the DL-unmix GAN method with the provided directories.

    Args:
        target_dir (str): The directory containing the target files.
        sc_dir (str): The directory containing the single-cell reference files.
        out_dir (str): The directory where the output will be saved.
        batch_size (int): The batch size for the GAN model.
        log_normalization (bool): Whether to perform log normalization on the input data.
        seed (int): The random seed for gene permutation.
    """
    print("Starting DL-unmix GAN")
    target_bulk = pd.read_csv(f"{target_dir}_pbs.tsv", sep="\t", index_col=0)
    target_frac = pd.read_csv(f"{target_dir}_frac.tsv", sep="\t",index_col=0)
    cell_types = np.sort(target_frac.columns.values)

    sc_count = pd.read_csv(f"{sc_dir}_count.tsv", sep="\t", index_col=0).T
    sc_meta = pd.read_csv(f"{sc_dir}_metadata.tsv", sep="\t", index_col=0)

    sc_meta.columns = ['cell_name', 'cell_type', 'donor']
    prefix = os.path.basename(sc_dir)
    sc_pbs, sc_cts, sc_frac = create_pb(sc_count, sc_meta,prefix=prefix, out_dir=os.path.dirname(sc_dir),qc_threshold=0.8)

    # Overlap genes between target and sc data
    common_genes = np.intersect1d(target_bulk.index, sc_pbs.index)
    target_bulk = target_bulk.loc[common_genes].T
    sc_pbs = sc_pbs.loc[common_genes].T
    sc_cts = sc_cts.loc[[f"{gene}_{ct}" for gene in common_genes for ct in cell_types]].T
    print(f"Number of common genes: {len(common_genes)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, 'to run the GAN model.')

    # Permutation
    original_genes = sc_pbs.columns.values
    permuted_genes = original_genes.copy() 
    seed = int(seed)
    if seed > 0:
        random.seed(seed)
        random.shuffle(permuted_genes)

        target_bulk = target_bulk[permuted_genes]
        sc_pbs = sc_pbs[permuted_genes]
        sc_cts = sc_cts[[f"{gene}_{ct}" for gene in permuted_genes for ct in cell_types]]

    # Initilize the output dataframe
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

    # Run the GAN model for each gene batch

    for b in range(n_batch):
        lower = b * batch_size
        upper = np.min([(b + 1) * batch_size, n_genes])
        print("Running the " + str(b + 1) + " batch.")

        genes = permuted_genes[lower:upper]
        print("Genes:", genes)

        # Assume lambda_consistency is defined (e.g., 1.0)
        lambda_consistency = 1.0

        # Prepare data as before:
        X = np.hstack([sc_pbs[genes].to_numpy(), sc_frac.to_numpy()])
        Y = sc_cts[[f"{gene}_{ct}" for gene in genes for ct in cell_types]].to_numpy()

        X2 = np.hstack([target_bulk[genes].to_numpy(), target_frac.to_numpy()])

        X_train = torch.FloatTensor(X).to(device)
        Y_train = torch.FloatTensor(Y).to(device)
        X_test = torch.FloatTensor(X2).to(device)

        # Setup Generator and Discriminator
        input_layer_size = X_train.shape[1]  # number of genes + celltypes
        output_layer_size = Y_train.shape[1]  # number of genes * celltypes
        hidden_layer_sizes = [8, 16, 64, input_layer_size]

        generator = Generator(input_layer_size, hidden_layer_sizes, output_layer_size).to(device)
        discriminator = Discriminator(output_layer_size, hidden_layer_sizes[:-1]).to(device)

        # Loss and optimizer
        adversarial_loss = nn.BCELoss()  # Binary Cross Entropy for discriminator
        reconstruction_loss = nn.MSELoss()  # Mean squared error for generator
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)

        real_label = torch.ones((X_train.size(0), 1)).to(device)
        fake_label = torch.zeros((X_train.size(0), 1)).to(device)

        B = upper - lower  # number of genes in the current batch

        for epoch in range(10000):
            # Train Generator
            optimizer_G.zero_grad()
            generated_data = generator(X_train)  # shape: (N, n_ct * B)
            
            # --- New Consistency Loss ---
            N = generated_data.size(0)
            # Reshape generator output to (N, n_ct, B)
            G = generated_data.view(N, n_ct, B)
            
            # Extract the bulk expression P and fractions F from X_train:
            # P: first B columns (bulk expression)
            # F: next n_ct columns (fractions)
            P = X_train[:, :B]             # shape: (N, B)
            F = X_train[:, B:]         # shape: (N, n_ct)
            
            # Compute weighted sum over cell types:
            weighted_sum = torch.sum(F.unsqueeze(2) * G, dim=1)  # shape: (N, B)
            consistency_loss = torch.nn.functional.mse_loss(weighted_sum, P)
            
            # Adversarial and reconstruction losses
            d_output_fake = discriminator(generated_data)
            adv_loss = adversarial_loss(d_output_fake, real_label)
            recon_loss = reconstruction_loss(generated_data, Y_train)
            
            # Total generator loss including consistency loss
            g_loss = adv_loss + recon_loss + lambda_consistency * consistency_loss
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_data_pred = discriminator(Y_train)
            fake_data_pred = discriminator(generated_data.detach())
            real_loss = adversarial_loss(real_data_pred, real_label)
            fake_loss = adversarial_loss(fake_data_pred, fake_label)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Adversarial Loss: {adv_loss.item()}, Reconstruction Loss: {recon_loss.item()}, "
                      f"Consistency Loss: {consistency_loss.item()}, Total Generator Loss: {g_loss.item()}, "
                      f"Discriminator Loss: {d_loss.item()}")

            # Early stopping check could be added here if needed

        y_pred = generator(X_test)
        y_pred[y_pred < 0] = 0  # Ensure non-negative predictions

        y_pred = y_pred.cpu().detach().numpy()
        pred[[f"{gene}_{ct}" for gene in genes for ct in cell_types]] = y_pred

    # Save the output
    pred = pred[[f"{gene}_{ct}" for gene in original_genes for ct in cell_types]]
    pred.to_csv(f"{out_dir}/gan_{batch_size}.tsv", sep="\t")
    print("DL-unmix GAN finished successfully.")
    return pred
