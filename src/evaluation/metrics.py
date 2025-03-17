import pandas as pd
import numpy as np
import re

def compute_ccc(x, y, axis=0):
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        if axis == 0:
            return x.apply(lambda col: compute_ccc(col, y[col.name]), axis=axis)
        else:
            return x.apply(lambda row: compute_ccc(row, y.loc[row.name]), axis=axis)
    else:
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x = np.var(x, ddof=1)
        var_y = np.var(y, ddof=1)
        covar = np.cov(x, y, ddof=1)[0, 1]
        denominator = var_x + var_y + (mean_x - mean_y) ** 2
        return 2 * covar / denominator if denominator != 0 else np.nan

def compute_deconvolution_accuracy(
    pred: pd.DataFrame or str,
    true: pd.DataFrame or str,
    out_dir: str,
    metric: str = 'pcc',
    log_normalization: list = [False, True],
    transpose: bool = [False, True],
    by_gene: bool = True
):
    """Computes the deconvolution accuracy between the predicted and true data.

    Args:
        pred (pd.DataFrame or str): The predicted data.
        true (pd.DataFrame or str): The true data.
        out_dir (str): The directory where the output will be saved.
        log_normalization (list): Whether to perform log normalization on the input data. Default is [False, True].
        transpose (list): Whether to transpose the input data. Default is [True, False].
        metric (str): The metric to use for computing the accuracy. Default is 'pcc'. Other options include 'ccc', 'mse'.
        If not specified, PCC will be used.
        by_gene (bool): Whether to compute the accuracy by gene. Default is True.
    """
    if isinstance(pred, str):
        pred = pd.read_csv(pred, sep="\t", index_col=0)
    if isinstance(true, str): 
        true = pd.read_csv(true, sep="\t", index_col=0)

    # Log normalization
    if log_normalization[0]:
        pred = np.log2(pred + 1)
    if log_normalization[1]:
        true = np.log2(true + 1)

    # Transpose
    if transpose[0]:
        pred = pred.T
    if transpose[1]:
        true = true.T

    # Make sure pred and true have string index and columns
    pred.index = pred.index.astype(str)
    pred.columns = pred.columns.astype(str)
    true.index = true.index.astype(str)
    true.columns = true.columns.astype(str)

    # Overlap data
    common_samples = pred.index.intersection(true.index)
    common_genes = pred.columns.intersection(true.columns)
    pred = pred.loc[common_samples, common_genes]
    true = true.loc[common_samples, common_genes]

    genes = sorted(set(re.sub(r'_.+', '', col) for col in pred.columns))
    cell_types = sorted(set(re.sub(r'.+_', '', col) for col in pred.columns))
    samples = pred.index

    if by_gene:
        score = pd.DataFrame(0, index=genes, columns=cell_types)
        cor_axis = 0
    else:
        score = pd.DataFrame(0, index=samples, columns=cell_types)
        cor_axis = 1
    for ct in cell_types:
        cols = [f"{gene}_{ct}" for gene in genes]
        pred_sub = pred.loc[:, cols]
        true_sub = true.loc[:, cols]
        if metric == 'pcc':
            corrs = pred_sub.corrwith(true_sub, axis=cor_axis)
            score.loc[:, ct] = corrs.values
        elif metric == 'mse':
            score.loc[:, ct] = ((pred_sub - true_sub) ** 2).mean(axis=cor_axis).values
        elif metric == 'ccc':
            score.loc[:, ct] = compute_ccc(pred_sub, true_sub, axis=cor_axis).values
    return score

