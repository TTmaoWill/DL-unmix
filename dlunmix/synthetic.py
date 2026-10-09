"""Small generated inputs for software demonstrations, not benchmark evidence."""
import numpy as np
import pandas as pd


def make_synthetic(seed=7, n_reference=16, n_target=5, n_genes=8, n_cell_types=3):
    rng = np.random.default_rng(seed)
    genes = [f"gene{g:03d}" for g in range(n_genes)]
    cell_types = [f"type{k}" for k in range(n_cell_types)]
    anchor = rng.normal(2.5, 0.7, size=(n_genes, n_cell_types))

    def cohort(n, prefix):
        donors = [f"{prefix}{i:03d}" for i in range(n)]
        fractions = rng.dirichlet(np.ones(n_cell_types)*2, size=n)
        expression = np.maximum(np.exp2(anchor[None] + rng.normal(0, 0.45, (n, n_genes, n_cell_types)))-1, 0)
        bulk = (expression * fractions[:, None]).sum(axis=2)
        return (pd.DataFrame(bulk, index=donors, columns=genes),
                pd.DataFrame(fractions, index=donors, columns=cell_types),
                {ct: pd.DataFrame(expression[:, :, k], index=donors, columns=genes) for k, ct in enumerate(cell_types)})

    return cohort(n_reference, "reference"), cohort(n_target, "target")
