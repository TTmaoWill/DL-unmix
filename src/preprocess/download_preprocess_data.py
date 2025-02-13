import urllib.request
import os
import gzip
import shutil
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import scanpy as sc
import gzip
import scipy.sparse as ss
import h5py
from multiprocessing import Pool
from functools import partial


def download_tasic_2018(out_dir: str = 'data/raw'):
    """
    Download Tasic et al. 2018 data from GEO.
    """
    # Download Tasic et al. 2018 exon counts data from GEO
    url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE115746&format=file&file=GSE115746%5Fcells%5Fexon%5Fcounts%2Ecsv%2Egz"
    out_file = os.path.join(dir,'tasic2018',"count_matrix.csv.gz")
    urllib.request.urlretrieve(url, out_file)

    # Unzip the file
    with gzip.open(out_file, 'rb') as f_in:
        with open(out_file.replace(".gz", ""), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove the gzipped file
    os.remove(out_file)

    # Download Tasic et al. 2018 metadata from GEO

    url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE115746&format=file&file=GSE115746%5Fcomplete%5Fmetadata%5F28706%2Dcells%2Ecsv%2Egz"
    out_file = os.path.join(dir,'tasic2018',"metadata.csv.gz")
    urllib.request.urlretrieve(url, out_file)

    # Unzip the file
    with gzip.open(out_file, 'rb') as f_in:
        with open(out_file.replace(".gz", ""), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(out_file)

def preprocess_tasic_2018(in_dir: str = 'data/raw', out_dir: str = 'data/processed', save_file: bool = True):
    """
    Preprocess Tasic et al. 2018 data.
    """
    # If raw data does not exist, download it
    if not os.path.exists(os.path.join(in_dir, 'tasic2018',"count_matrix.csv")):
        download_tasic_2018(in_dir)
    # If processed data exists, load it
    if os.path.exists(os.path.join(out_dir, "tasic2018_count.tsv")):
        return pd.read_csv(os.path.join(out_dir, "tasic2018_count.tsv"), sep='\t'), pd.read_csv(os.path.join(out_dir, "tasic2018_metadata.tsv"), sep='\t')
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
    if save_file:
        subset.to_csv(os.path.join(out_dir, "tasic2018_count.tsv"), sep='\t', index=False)
        meta.to_csv(os.path.join(out_dir, "tasic2018_metadata.tsv"), sep='\t',index=False)

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



def preprocess_yao_2021(in_dir: str = 'data/raw', out_dir: str = 'data/processed', save_file: bool = True):
    """
    Preprocess Yao et al. 2021 data.
    """
    # If raw data does not exist, download it
    f_name = os.path.join(in_dir, 'yao2021', "smrt.h5")
    if not os.path.exists(f_name):
        download_yao_2021(in_dir)
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

    # Load metadata
    # Load metadata from tar file
    tar_file = os.path.join(in_dir, 'yao2021', 'CTX_Hip_anno_SSv4.csv.tar')
    meta = pd.read_csv(tar_file, compression='tar')

    

    # Save the preprocessed data if requested

    return exons

