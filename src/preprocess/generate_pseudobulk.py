import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm

def create_pb(
    expression_matrix,
    metafile=None,
    gene_names=None,
    cell_names=None,
    cell_types=None,
    donors=None,
    mode="by-id",
    n_cells=None,
    frac_prior=None,
    qc_threshold=0,
    out_dir="output",
):
    """
    Generate pseudobulk matrices: pbs, cts, frac.

    Args:
        expression_matrix (pd.DataFrame): Gene x Cell expression matrix.
        metafile (pd.DataFrame, optional): Metadata with columns ['cell_name', 'cell_type', 'donor'].
        gene_names (str, optional): Column name or index for gene names in the expression matrix.
        cell_names (str, optional): Column name or index for cell names in the expression matrix.
        cell_types (list, optional): List of cell types for each cell if metafile is not provided.
        donors (list, optional): List of donor IDs for each cell if metafile is not provided.
        mode (str): Either 'by-id' or 'random'.
        n_cells (list): Vector of length equal to the final pseudobulk count, with cell numbers per pseudobulk.
        frac_prior (pd.DataFrame, optional): A #pb x celltype matrix for cell type fractions.
        qc_threshold (float): Threshold to filter genes expressed in less than (1-qc_threshold) fraction of pseudobulks.
        out_dir (str): Directory to save output files.

    Returns:
        None: Outputs are saved as TSV files in out_dir.
    """
    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Validate input dimensions
    if not isinstance(expression_matrix, pd.DataFrame):
        raise ValueError("expression_matrix must be a pandas DataFrame.")

    # Assign gene and cell names if provided
    if gene_names:
        expression_matrix.index = expression_matrix[gene_names]
    if cell_names:
        expression_matrix.columns = expression_matrix[cell_names]

    # Construct metafile if not provided
    if metafile is None:
        if cell_names is None or cell_types is None or donors is None:
            raise ValueError("If metafile is not provided, cell_names, cell_types, and donors must be specified.")
        metafile = pd.DataFrame({
            "cell_name": expression_matrix.columns,
            "cell_type": cell_types,
            "donor": donors
        })

    # Merge expression_matrix with metadata
    merged = expression_matrix.T.join(metafile.set_index("cell_name"), how="inner")
    print(f"Matched data: {merged.shape[0]} cells, {len(merged['donor'].unique())} donors, {expression_matrix.shape[0]} genes.")

    # Check donor presence for by-id mode
    if mode == "by-id" and "donor" not in metafile.columns:
        raise ValueError("Donor information is required for 'by-id' mode.")

    # Initialize outputs
    pbs = []
    cts = []
    frac = []

    # Generate pseudobulk samples
    unique_cell_types = metafile["cell_type"].unique()

    for idx, n in enumerate(tqdm(n_cells, desc="Generating pseudobulks")):
        if mode == "by-id":
            # Group cells by donor
            donors = merged.groupby("donor")
            selected_donor = random.choice(list(donors.groups.keys()))
            sampled_cells = donors.get_group(selected_donor)
        elif mode == "random":
            if frac_prior is None:
                raise ValueError("frac_prior must be provided for mode 'random'.")
            # Stratify sampling by cell type if frac_prior is provided
            cell_type_fractions = frac_prior.iloc[idx]
            sampled_cells = []
            for cell_type, fraction in cell_type_fractions.items():
                cell_type_group = merged[merged["cell_type"] == cell_type]
                n_type_cells = int(fraction * n)
                sampled_cells.append(cell_type_group.sample(n=n_type_cells, replace=True))
            sampled_cells = pd.concat(sampled_cells)
        else:
            raise ValueError("mode must be 'by-id' or 'random'")

        # Aggregate pseudobulk expression
        sample_pb = sampled_cells.drop(columns=["cell_type", "donor"]).sum()
        pbs.append(sample_pb)

        # Aggregate by cell type
        sample_cts = sampled_cells.groupby("cell_type").sum().add_prefix(f"sample{idx}_")
        cts.append(sample_cts)

        # Calculate cell type fractions
        cell_type_counts = sampled_cells["cell_type"].value_counts(normalize=True)
        frac.append(cell_type_counts)

    # Create final DataFrames
    pbs_df = pd.DataFrame(pbs).T
    cts_df = pd.concat(cts, axis=1).T
    frac_df = pd.DataFrame(frac).fillna(0).T

    # Apply QC threshold if specified
    if qc_threshold > 0:
        gene_fractions = (pbs_df > 0).sum(axis=1) / pbs_df.shape[1]
        filtered_genes = gene_fractions[gene_fractions >= (1 - qc_threshold)].index
        print(f"After QC, {len(filtered_genes)} genes remain.")
        pbs_df = pbs_df.loc[filtered_genes]
        cts_df = cts_df.loc[filtered_genes]

    # Save outputs to TSV files
    pbs_df.to_csv(os.path.join(out_dir, "pb.txt"), sep="\t")
    cts_df.to_csv(os.path.join(out_dir, "cts.txt"), sep="\t")
    frac_df.to_csv(os.path.join(out_dir, "frac.txt"), sep="\t")

    print(f"Pseudobulk files saved in {out_dir}")
