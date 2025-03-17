import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

def plot_scores(score_dir, out_dir, metric):
    """
    Plots violin plots of scores for each cell type, stratified by method,
    arranged in a grid layout.

    Args:
        score_dir (str): Directory containing the score files (TSV).
        out_dir (str): Directory to save the plots.
        metric (str): The metric to plot (e.g., 'accuracy', 'f1').
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = {}
    for filename in os.listdir(score_dir):
        if filename.endswith(f"{metric}.tsv"):
            method = filename.split(f"_{metric}.tsv")[0]
            filepath = os.path.join(score_dir, filename)
            data[method] = pd.read_csv(filepath, sep='\t', index_col=0)

    cell_types = data[list(data.keys())[0]].columns  # Assuming all files have the same cell types
    methods = list(data.keys())

    # Calculate grid dimensions for a near-square layout
    num_cell_types = len(cell_types)
    n_cols = math.ceil(math.sqrt(num_cell_types))
    n_rows = math.ceil(num_cell_types / n_cols)
    
    # Create a color palette for the methods
    palette = sns.color_palette("colorblind", len(methods))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten() if num_cell_types > 1 else [axes]

    for i, cell_type in enumerate(cell_types):
        if i >= len(axes):
            break
            
        plot_data = []
        for j, (method, df) in enumerate(data.items()):
            scores = df[cell_type].tolist()
            method_data = pd.DataFrame({
                'Method': [method] * len(scores),
                'Score': scores
            })
            plot_data.append(method_data)
        
        plot_df = pd.concat(plot_data, ignore_index=True)
        
        # Create violin plot with colors by method
        vp = sns.violinplot(x='Method', y='Score', data=plot_df, ax=axes[i], 
                        palette=palette, cut=0, inner=None)
        
        # Add mean markers with smaller size
        for violin_idx, (violin, method) in enumerate(zip(vp.collections[::2], methods)):
            # Calculate mean for this method and cell type
            mean_val = plot_df[plot_df['Method'] == method]['Score'].mean()
            # Plot mean point - safely handle case where violin has no paths
            if violin.get_paths() and len(violin.get_paths()) > 0:
                x_pos = violin.get_paths()[0].vertices[:, 0].mean()
            else:
                # If violin has no path, use the x-position based on the violin index
                x_pos = violin_idx
            # Using smaller marker size (15 instead of 30)
            axes[i].scatter(x_pos, mean_val, color='white', s=15, zorder=3, edgecolor='black', linewidth=0.5)
        
        axes[i].set_title(f'{cell_type}')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        
    # Hide unused subplots
    for j in range(num_cell_types, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'{metric.capitalize()} Scores by Cell Type', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(os.path.join(out_dir, f'all_cell_types_{metric}_grid.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot scores from TSV files.')
    parser.add_argument('--score_dir', type=str, help='Directory containing the score files.')
    parser.add_argument('--out_dir', type=str, help='Directory to save the plots.')
    parser.add_argument('--metric', type=str, help='The metric to plot (e.g., accuracy, f1).')

    args = parser.parse_args()

    plot_scores(args.score_dir, args.out_dir, args.metric)
