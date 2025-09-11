import urllib.request
import os
import gzip
import shutil
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import scipy.sparse as ss
import h5py
from multiprocessing import Pool
from functools import partial
import scanpy as sc

def download_tasic_2018(out_dir: str = 'data/raw'):
    """
    Download Tasic et al. 2018 data from GEO.
    """
    # Download Tasic et al. 2018 exon counts data from GEO
    url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE115746&format=file&file=GSE115746%5Fcells%5Fexon%5Fcounts%2Ecsv%2Egz"
    out_file = os.path.join(out_dir,'tasic2018',"count_matrix.csv.gz")
    urllib.request.urlretrieve(url, out_file)

    # Unzip the file
    with gzip.open(out_file, 'rb') as f_in:
        with open(out_file.replace(".gz", ""), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove the gzipped file
    os.remove(out_file)

    # Download Tasic et al. 2018 metadata from GEO

    url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE115746&format=file&file=GSE115746%5Fcomplete%5Fmetadata%5F28706%2Dcells%2Ecsv%2Egz"
    out_file = os.path.join(out_dir,'tasic2018',"metadata.csv.gz")
    urllib.request.urlretrieve(url, out_file)

    # Unzip the file
    with gzip.open(out_file, 'rb') as f_in:
        with open(out_file.replace(".gz", ""), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(out_file)

def preprocess_tasic_2018(in_dir: str = 'data/raw',
    out_dir: str = 'data/processed/mouse_brain',
    overwrite: bool = False):
    """
    Preprocess Tasic et al. 2018 data.
    """
    # If raw data does not exist, download it
    if not os.path.exists(os.path.join(in_dir, 'tasic2018',"count_matrix.csv")):
        download_tasic_2018(in_dir)
    # If processed data exists, load it
    if os.path.exists(os.path.join(out_dir, "tasic2018_count.tsv")) and not overwrite:
        return pd.read_csv(os.path.join(out_dir, "tasic2018_count.tsv"), sep='\t',index_col=0), pd.read_csv(os.path.join(out_dir, "tasic2018_metadata.tsv"), sep='\t')
    # Load the data
    data = pd.concat([chunk for chunk in tqdm(pd.read_csv(os.path.join(in_dir,'tasic2018',"count_matrix.csv"), chunksize=1000, index_col=0), desc='Loading data')])
    
    meta = pd.read_csv(os.path.join(in_dir,'tasic2018',"metadata.csv"))
    
    meta = meta[meta['cell_subclass'].isin(['Astro', 'Oligo', 'L2/3 IT', 'L6 CT', 'L6 IT', 'Lamp5', 'Pvalb', 'Sst', 'Vip'])]

    common_cells = data.columns.intersection(meta['sample_name'])
    
    # Subset gene matrix
    subset = data.loc[:, common_cells]

    # Filter genes not expressed in at least 10 cells
    subset = subset.loc[subset.sum(axis=1) >= 10, :]

    # Filter genes with low variance
    subset = subset.loc[subset.var(axis=1) >= 1, :]

    # Rename to meet pb generation requirement
    meta = meta[meta['sample_name'].isin(common_cells)]
    meta = meta.loc[:,['sample_name', 'cell_subclass','donor_id']]
    meta.columns = ['cell_name', 'cell_type', 'donor']

    # Save the data
    subset.to_csv(os.path.join(out_dir, "tasic2018_count.tsv"), sep='\t', index=True)
    meta.to_csv(os.path.join(out_dir, "tasic2018_metadata.tsv"), sep='\t',index=False)
    subset.index.to_series().to_csv(os.path.join(out_dir, "tasic2018_genes.tsv"), sep='\t', index=False)

    return subset, meta


def download_yao_2021(out_dir: str = 'data/raw'):
    """
    Download Yao et al. 2021 data 
    """
    # Download Yao et al. 2021 data from nemo
    base_url = "https://data.nemoarchive.org/biccn/grant/u19_zeng/zeng/transcriptome/scell/SSv4/mouse/processed/YaoHippo2020/"

    # Create output directory
    os.makedirs(os.path.join(out_dir, 'yao2021'), exist_ok=True)

    # Get list of files from webpage
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    files = [link.get('href') for link in soup.find_all('a') if '.' in link.get('href')]

    # Download each file
    for file in files:
        url = base_url + file
        out_file = os.path.join(out_dir, 'yao2021', file)
        urllib.request.urlretrieve(url, out_file)



def preprocess_yao_2021(in_dir: str = 'data/raw',
    out_dir: str = 'data/processed/mouse_brain',
    overwrite: bool = False):
    """
    Preprocess Yao et al. 2021 data.
    """
    f_name = os.path.join(in_dir, 'yao2021', "smrt.h5")
    # If raw data does not exist, download it
    if not os.path.exists(f_name):
        download_yao_2021(in_dir)
    # If processed data exists, load it
    if os.path.exists(os.path.join(out_dir, "yao2021_count.tsv")) and not overwrite:
        return pd.read_csv(os.path.join(out_dir, "yao2021_count.tsv"), sep='\t',index_col=0), pd.read_csv(os.path.join(out_dir, "yao2021_metadata.tsv"), sep='\t')
    def extract_sparse_mat(h5f, data_path):
        print(f"Extracting data from {data_path}")
        data = h5f[data_path]
        # Read all data at once into memory
        x = data['x'][:]  # data values
        i = data['i'][:]  # row indices
        p = data['p'][:]  # column pointers
        dims = data['dims'][:]  # matrix dimensions
        # Directly construct CSC matrix using the h5 file format
        # which is already in CSC format (Compressed Sparse Column)
        print("Constructing sparse matrix...")
        sparse_mat = ss.csc_matrix((x, i, p), shape=(dims[0], dims[1]))
        return sparse_mat
    
    print("Opening HDF5 file...")
    h5f = h5py.File(f_name, 'r')
    exons = extract_sparse_mat(h5f, '/data/exon/')

    genes = h5f['gene_names'][:].astype(str)
    samples = h5f['sample_names'][:].astype(str)
    # Create DataFrame with genes as index and samples as columns
    df = pd.DataFrame.sparse.from_spmatrix(exons, index=samples, columns=genes)

    # Load metadata
    # Load metadata from tar file
    tar_file = os.path.join(in_dir, 'yao2021', 'CTX_Hip_anno_SSv4.csv.tar')
    meta = pd.read_csv(tar_file, compression='tar')

    # Remove cells included in Tasic2018
    meta = meta[meta['tasic18_subclass_label'] == 'absent']
    donor_counts = meta['donor_label'].value_counts()

    # Filter for cell types of interest and rename them to match Tasic2018 naming
    class_mapping = {
        'L2/3 IT CTX': 'L2/3 IT',
        'L2/3 IT PPP': 'L2/3 IT',
        'L2/3 IT ENTl': 'L2/3 IT',
        'L2/3 IT RHP': 'L2/3 IT',
        'L2 IT ENTl': 'L2/3 IT',
        'L2 IT ENTm': 'L2/3 IT',
        'L6 CT CTX': 'L6 CT',
        'L6 IT CTX': 'L6 IT',
        'L6 IT ENTl': 'L6 IT',
        'Astro': 'Astro',
        'Oligo': 'Oligo',
        'Lamp5': 'Lamp5',
        'Pvalb': 'Pvalb',
        'Sst': 'Sst',
        'Vip': 'Vip'
    }

    # Apply mapping and filter
    meta['cell_subclass'] = meta['subclass_label'].map(class_mapping)
    meta = meta.dropna(subset=['cell_subclass'])

    # Get common cells between metadata and expression data
    common_cells = df.index.intersection(meta['exp_component_name'])

    # Subset both datasets
    meta = meta[meta['exp_component_name'].isin(common_cells)]
    df = df.loc[common_cells]

    # Filter genes included in Tasic2018
    tasic_genes = pd.read_csv(os.path.join(out_dir, "tasic2018_genes.tsv"), sep='\t')['0']
    df = df.loc[:, tasic_genes]
    
    df = df.sparse.to_dense().astype(int)

    # Rename columns to match requirements
    meta = meta.loc[:, ['exp_component_name', 'cell_subclass', 'donor_label']]
    meta = meta[meta['exp_component_name'].isin(df.index)]
    meta.columns = ['cell_name', 'cell_type', 'donor']

    df.to_csv(os.path.join(out_dir, "yao2021_count.tsv"), sep='\t')
    meta.to_csv(os.path.join(out_dir, "yao2021_metadata.tsv"), sep='\t', index=False)

    return df, meta


def download_yazar_2022(out_dir: str = 'data/raw'):
    """
    Preprocess Yazar et al. 2022 data.
    """
    
    # Create dir to save file
    dl_out_file = os.path.join(out_dir,'yazar2022',"count_adata.h5ad")
    dir_path = os.path.dirname(dl_out_file)
    os.makedirs(dir_path, exist_ok=True)

    # Download Yazar 2022 H5AD data
    url = "https://datasets.cellxgene.cziscience.com/81d84489-bff9-4fb6-b0ee-78348126eada.h5ad"
    urllib.request.urlretrieve(url, dl_out_file)
    
def preprocess_yazar_2022(in_dir: str = 'data/raw',
    out_dir: str = 'data/processed/yazar2022',
    overwrite: bool = False):
    
    """
    Preprocess Yazar et al. 2022 data.
    """
        
    # If processed data exists, load it
    if os.path.exists(os.path.join(out_dir, "yazar2022_count.tsv")) and not overwrite:
        return pd.read_csv(os.path.join(out_dir, "yazar2022_count.tsv"), sep='\t',index_col=0), \
               pd.read_csv(os.path.join(out_dir, "yazar2022_metadata.tsv"), sep='\t')
    
    # If raw data does not exist, download it
    if not os.path.exists(os.path.join(in_dir, 'yazar2022', "count_adata.h5ad")):
        print("Downloading data", flush=True)
        download_yazar_2022(in_dir)
        
    # Load the data
    print("Reading data", flush=True)
    data = sc.read_h5ad(os.path.join(in_dir,'yazar2022',"count_adata.h5ad"))
    
    print(data, flush=True)

    print("Preprocessing data", flush=True)
    # Calculate QC metrics
    data.var['mt'] = data.var_names.str.startswith('MT-')  # Mitochondrial genes (for human)
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Filter cells and genes
    data = data[data.obs.n_genes_by_counts < 2500, :]
    data = data[data.obs.pct_counts_mt < 5, :]
    data = data[:, data.var.n_cells_by_counts > 10]

    # Transpose
    data = data.copy().T  # gene x samples
    
    # Extract and save metadata
    os.makedirs(out_dir, exist_ok=True)
    meta = data.var.reset_index() # Samples Data
    meta = meta.rename({"barcode":"cell_name", "donor_id":"donor"},axis=1)
    meta = meta[["cell_name", "cell_type", "donor"]]
    meta.to_csv(os.path.join(out_dir, "yazar2022_metadata.tsv"), sep='\t')
    data.obs.index.to_series().to_csv(os.path.join(out_dir, "yazar2022_genes.tsv"), sep='\t', index=False) # Genes

    # Extract raw counts and save
    data_df = pd.DataFrame.sparse.from_spmatrix(data.X, index=data.obs.index, columns=data.var.index)
    data_df = data_df.sparse.to_dense()
    data_df = data_df.loc[data_df.var(axis=1) >= 1, :]
    data_df.to_csv(os.path.join(out_dir, "yazar2022_count.tsv"), sep='\t')

    # Remove original H5AD
    os.remove(os.path.join(in_dir,'yazar2022',"count_adata.h5ad"))

    return data_df, meta

def download_aida_data(out_dir: str = 'data/raw'):
    """
    Preprocess AIDA data.
    """
    
    # Create dir to save file
    dl_out_file = os.path.join(out_dir,'aida',"count_adata.h5ad")
    dir_path = os.path.dirname(dl_out_file)
    os.makedirs(dir_path, exist_ok=True)

    # Download AIDA H5AD data
    url = "https://datasets.cellxgene.cziscience.com/0fce5dd5-bcec-4288-90b3-19a16b45ad16.h5ad"
    urllib.request.urlretrieve(url, dl_out_file)


def preprocess_aida_data(in_dir: str = 'data/raw',
    out_dir: str = 'data/processed/aida',
    overwrite: bool = False):
    
    """
    Preprocess AIDA et al. data.
    """
            
    # If processed data exists, load it
    if os.path.exists(os.path.join(out_dir, "aida_count.tsv")) and not overwrite:
        print("Reading processed data", flush=True)
        return pd.read_csv(os.path.join(out_dir, "aida_count.tsv"), sep='\t',index_col=0), \
               pd.read_csv(os.path.join(out_dir, "aida_metadata.tsv"), sep='\t')
    
    # If raw data does not exist, download it
    if not os.path.exists(os.path.join(in_dir, 'aida', "count_adata.h5ad")):
        print("Downloading data", flush=True)
        download_aida_data(in_dir)

    # Load the data
    print("Reading raw data", flush=True)
    data = sc.read_h5ad(os.path.join(in_dir,'aida',"count_adata.h5ad"))
    
    print(data, flush=True)

    print("Preprocessing data", flush=True)
    # Calculate QC metrics
    data.var['mt'] = data.var_names.str.startswith('MT-')  # Mitochondrial genes (for human)
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Filter cells and genes
    data = data[data.obs.n_genes_by_counts < 2500, :]
    data = data[data.obs.pct_counts_mt < 5, :]
    data = data[:, data.var.n_cells_by_counts > 10]

    print(data, flush=True)

    # Transpose
    data = data.copy().T  # gene x samples

    # Extract and save metadata
    os.makedirs(out_dir, exist_ok=True)
    meta = data.var # Samples Data
    meta = meta.reset_index(names='cell_name')
    meta = meta.rename({"donor_id":"donor"},axis=1)
    meta = meta[["cell_name", "cell_type", "donor"]]
    meta.to_csv(os.path.join(out_dir, "aida_metadata.tsv"), sep='\t', index=False)
    data.obs.index.to_series().to_csv(os.path.join(out_dir, "aida_genes.tsv"), sep='\t', index=False) # Genes

    # Extract raw counts and save
    data_df = pd.DataFrame.sparse.from_spmatrix(data.X, index=data.obs.index, columns=data.var.index)
    data_df = data_df.sparse.to_dense()
    data_df = data_df.loc[data_df.var(axis=1) >= 1, :]
    data_df.to_csv(os.path.join(out_dir, "aida_count.tsv"), sep='\t')

    # Remove original H5AD
    os.remove(os.path.join(in_dir,'aida',"count_adata.h5ad"))

    print("Done preprocessing data", flush=True)

    return data_df, meta