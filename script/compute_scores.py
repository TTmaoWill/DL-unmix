import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from src.evaluation.metrics import *

#!/usr/bin/env python
# -*- coding: utf-8 -*-


def parse_arguments():
    parser = argparse.ArgumentParser(description='Compute evaluation metrics between predicted and true values.')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing prediction TSV files')
    parser.add_argument('--true_file', type=str, required=True, help='Path to ground truth TSV file')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for evaluation results')
    parser.add_argument('--metric', type=str, default='pcc', help='Metric to compute (default: pcc)')
    
    # Parse known args first
    args, unknown = parser.parse_known_args()
    
    # Add any additional arguments that might be needed for the metrics function
    for i in range(0, len(unknown), 2):
        if i+1 < len(unknown):
            key = unknown[i].lstrip('-')
            value = unknown[i+1]
            setattr(args, key, value)
    
    return args

def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Get all TSV files in pred_dir
    pred_files = [f for f in os.listdir(args.pred_dir) if f.endswith('.tsv')]
    
    # Load the true file
    true_df = pd.read_csv(args.true_file, sep='\t',index_col=0)
    
    for pred_file in pred_files:
        print(f"Evaluating file: {pred_file}")
        
        # Extract prefix from prediction filename
        prefix = Path(pred_file).stem
        if '.' in prefix:  # Handle cases like "prefix.something.tsv"
            prefix = prefix[:prefix.rindex('.')]
        
        # Load prediction data
        pred_path = os.path.join(args.pred_dir, pred_file)
        pred_df = pd.read_csv(pred_path, sep='\t', index_col=0)
        
        # Create a dictionary of additional arguments to pass to the metrics function
        metric_args = vars(args).copy()
        del metric_args['pred_dir']
        del metric_args['true_file']
        del metric_args['out_dir']
        del metric_args['metric']
        
        print(f"Computing {args.metric} for {pred_file}...")
        result_df = compute_deconvolution_accuracy(pred_df, true_df, args.out_dir, args.metric, **metric_args)
        
        # Save the result
        out_file = f"{prefix}_{args.metric}.tsv"
        out_path = os.path.join(args.out_dir, out_file)
        result_df.to_csv(out_path, sep='\t')
        print(f"Saved results to {out_path}")

if __name__ == '__main__':
    main()