import sys
import importlib
import os
import numpy as np
import src.deconvolution
import src.preprocess
from src.preprocess.download_preprocess_data import *
from src.preprocess.generate_pseudobulk import *

def parse_key_value_args(args):
    kwargs = {}
    for arg in args:
        if '=' in arg:
            key, value_str = arg.split('=', 1)
            key = key.lstrip('--')
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str
            kwargs[key] = value
    return kwargs

def main():
    if len(sys.argv) < 2:
        raise ValueError("Method not specified. Please specify the method as the first argument (bmind or gan).")

    kwargs = parse_key_value_args(sys.argv[1:])
    method = kwargs['method']
    del kwargs['method']

    # (Download) Preprocess Tasic et al. 2018 data
    if not os.path.exists("data/processed/mouse_brain/tasic2018_common_genes_pbs.tsv"):
        target_count, target_meta = preprocess_tasic_2018()

        # Generate pseudo-bulk data using Tasic et al. 2018 data
        target_pbs, target_cts, target_frac = create_pb(
            target_count,
            target_meta,
            out_dir="data/processed/mouse_brain",
            prefix="tasic2018",
            qc_threshold=0.8,
            force=True
        )

    if not os.path.exists("data/processed/mouse_brain/yao2021_common_genes_count.tsv"):
        # (Download) Preprocess Yao et al. 2021 data
        scref_count, scref_meta = preprocess_yao_2021()

        common_genes = np.intersect1d(target_pbs.index, scref_count.columns)
        scref_count.loc[:, common_genes].to_csv("data/processed/mouse_brain/yao2021_common_genes_count.tsv", sep="\t", index=True)
        scref_meta.to_csv("data/processed/mouse_brain/yao2021_common_genes_meta.tsv", sep="\t", index=True)

        save_pbs(target_pbs, target_cts, target_frac,
               out_dir="data/processed/mouse_brain",
               prefix="tasic2018_common_genes",
               gene_list=common_genes)

    src.deconvolution.utils.run_deconvolution(
        method,
        target_dir='data/processed/mouse_brain/tasic2018_common_genes',
        sc_dir='data/processed/mouse_brain/yao2021_common_genes',
        out_dir="data/results/mouse_brain",
        **kwargs
        )

if __name__ == "__main__":
    main()