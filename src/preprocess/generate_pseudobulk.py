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
	out_dir='',
	prefix='',
	force=False,
	gene_list=None
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
	# Ensure output directory exists:
	if out_dir != '':
		os.makedirs(out_dir, exist_ok=True)
		# If output exists, just run get_pb
		if os.path.exists(f"{out_dir}/{prefix}_pbs.tsv") and not force:
			print(f"Output files already exist in {out_dir}. Loading existing files.")
			return get_pb(prefix, out_dir)
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
	# Set n_cells for by-id mode
	if mode == "by-id":
		donors = merged["donor"].unique()
		n_cells = merged["donor"].value_counts().loc[donors].values
	else:
		donors = range(len(n_cells))
	# Initialize outputs
	pbs = cts = frac = pd.DataFrame()
	# Generate pseudobulk samples
	unique_cell_types = np.sort(metafile["cell_type"].unique())
	
	for idx, (donor, n) in enumerate(tqdm(zip(donors, n_cells), desc="Generating pseudobulks", total=len(donors))):
		if mode == "by-id":
			sampled_cells = merged[merged["donor"] == donor]
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
		# Aggregate pseudobulk expression (genes x samples)
		sample_pb = sampled_cells.drop(columns=["cell_type", "donor"]).sum().T
		sample_pb.name = f"{donor}"
		pbs = pd.concat([pbs, sample_pb], axis=1)
		# Aggregate by cell type (genes x cell_types)
		sample_cts = sampled_cells.groupby("cell_type").sum().drop(columns=["donor"])
		for cell_type in unique_cell_types:
			if cell_type not in sample_cts.index:
				sample_cts.loc[cell_type,:] = 0
		sample_cts = sample_cts.sort_index()
		sample_cts_flat = pd.Series(sample_cts.to_numpy().flatten(order='F'))
		sample_cts_flat.index = [f"{gene}_{ct}" for gene in sample_cts.columns for ct in sample_cts.index]
		sample_cts_flat.name = f"{donor}"
		cts = pd.concat([cts, sample_cts_flat], axis=1)
		# Calculate cell type fractions (samples x cell_types)
		cell_type_counts = sampled_cells["cell_type"].value_counts(normalize=True)
		cell_type_fractions = pd.Series(0, index=unique_cell_types)
		cell_type_fractions.update(cell_type_counts)
		cell_type_fractions.name = f"{donor}"
		frac = pd.concat([frac, cell_type_fractions], axis=1)

	# filter genes expressed in less than qc_threshold fraction of pseudobulks
	frac = frac.T
	print(pbs.shape)
	print((pbs > 0).mean())
	if qc_threshold > 0:
		gene_qc = pbs.index[(pbs > 0).mean(axis=1) > qc_threshold]
		print(gene_qc)
		pbs = pbs.loc[gene_qc]
		cts = cts.loc[[f"{gene}_{ct}" for gene in gene_qc for ct in unique_cell_types]]
			# print out the number of genes after filtering
		print(f"Number of genes after QC: {len(gene_qc)}")

	# Save outputs
	if out_dir:
		save_pbs(pbs, cts, frac, out_dir, prefix)

	return pbs, cts, frac

def get_pb(prefix, out_dir="output"):
	"""
	Load pseudobulk matrices: pbs, cts, frac.

	Args:
		prefix (str): Prefix for pseudobulk files.
		out_dir (str): Directory to load pseudobulk files from.

	Returns:
		pd.DataFrame: Pseudobulk expression matrix.
		pd.DataFrame: Pseudobulk cell type matrix.
		pd.DataFrame: Pseudobulk cell type fraction matrix.
	"""
	pbs = pd.read_csv(f"{out_dir}/{prefix}_pbs.tsv", sep="\t", index_col=0)
	cts = pd.read_csv(f"{out_dir}/{prefix}_cts.tsv", sep="\t", index_col=0)
	frac = pd.read_csv(f"{out_dir}/{prefix}_frac.tsv", sep="\t", index_col=0)
	print(f"Number of genes after QC: {pbs.shape[0]}")
	return pbs, cts, frac

def save_pbs(pbs, cts, frac, out_dir="output", prefix="",gene_list=None):
	"""
	Save pseudobulk matrices: pbs, cts, frac.

	Args:
		pbs (pd.DataFrame): Pseudobulk expression matrix.
		cts (pd.DataFrame): Pseudobulk cell type matrix.
		frac (pd.DataFrame): Pseudobulk cell type fraction matrix.
		out_dir (str): Directory to save pseudobulk files.
		prefix (str): Prefix for pseudobulk files.

	Returns:
		None: Outputs are saved as TSV files in out_dir.
	"""
	os.makedirs(out_dir, exist_ok=True)
	cell_types = np.sort(frac.columns.values)
	if gene_list is not None:
		pbs = pbs.loc[gene_list]
		cts = cts.loc[[f"{gene}_{ct}" for gene in gene_list for ct in cell_types]]
	pbs.to_csv(f"{out_dir}/{prefix}_pbs.tsv", sep="\t")
	cts.to_csv(f"{out_dir}/{prefix}_cts.tsv", sep="\t")
	frac.to_csv(f"{out_dir}/{prefix}_frac.tsv", sep="\t")