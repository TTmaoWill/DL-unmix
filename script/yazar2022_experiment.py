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

    # (Download) Preprocess Yazar 2022 data
    if not os.path.exists("data/processed/yazar2022/yazar2022_pbs.tsv"):
        target_count, target_meta = preprocess_yazar_2022()

        print("Pseudobulk work", flush=True)
        # Generate pseudo-bulk data using Tasic et al. 2018 data
        target_pbs, target_cts, target_frac = create_pb(
            target_count,
            target_meta,
            out_dir="data/processed/yazar2022",
            prefix="yazar2022",
            qc_threshold=0.8,
            force=True
        )

    src.deconvolution.utils.run_deconvolution(
        method,
        target_dir='data/processed/yazar2022/yazar2022',
        sc_dir='data/processed/yazar2022/yazar2022',
        out_dir="data/results/yazar2022",
        **kwargs
        )

if __name__ == "__main__":
    main()