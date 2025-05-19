import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

# Activate the automatic conversion of pandas objects to R objects
pandas2ri.activate()

def run_bMIND(target_dir: str, sc_dir: str, out_dir: str):
    """Runs the bMIND R function with the provided directories.

    Args:
        target_dir (str): The directory containing the target files.
        sc_dir (str): The directory containing the single-cell reference files.
        out_dir (str): The directory where the output will be saved.
    """
    print("Starting bMIND")

    # Define the R function
    r_code = """
    function(target_dir, sc_dir, out_dir) {
        lib_path <- '/proj/yunligrp/users/djpharr/library'
        .libPaths(lib_path)

        library(MIND)
        library(tidyverse)
        library(data.table)
        bulk <- read_tsv(paste0(target_dir, '_pbs.tsv'), col_names = TRUE) %>%
            column_to_rownames('...1') 
        frac <- read_tsv(paste0(target_dir, '_frac.tsv'), col_names = TRUE) %>%
        column_to_rownames('...1')
        sc_ref <- read_tsv(paste0(sc_dir, '_count.tsv'),col_names = TRUE) %>%
            column_to_rownames('...1')
        sc_meta <- read_tsv(paste0(sc_dir, '_metadata.tsv'), col_names = TRUE) %>%
                    column_to_rownames(var = colnames(.)[1])
        sc_ref <- t(sc_ref)
    
        colnames(sc_meta) <- c('sample_name', 'cell_type', 'sample')
    
        index <- intersect(colnames(sc_ref), sc_meta$sample_name)
        sc_ref <- sc_ref[, index]

        sc_meta <- sc_meta %>% dplyr::filter(sample_name %in% index) %>%
            as.matrix() %>%
            as.data.frame() %>%
            `colnames<-`(c('sample_name', 'cell_type', 'sample'))
    
        bulk = bulk[order(rownames(bulk)), ]
        sc_ref = sc_ref[order(rownames(sc_ref)), ]

        print(sc_ref[1:5, 1:5])
        print(head(sc_meta))
    
        prior = get_prior(sc_ref, meta_sc = sc_meta)
    
        bulk_sub <- bulk[rownames(bulk) %in% rownames(prior$profile), ]
        bulk_sub <- bulk_sub[, order(colnames(bulk_sub))]
    
        frac <- frac[, order(colnames(frac))]
        celltype <- colnames(frac)
        colnames(frac) = paste0('c', 1:ncol(frac))
        dimnames(prior$cov)[[2]] = colnames(frac)
        dimnames(prior$cov)[[3]] = colnames(frac)
        colnames(prior$profile) = colnames(frac)
    
        posterior = bMIND(bulk_sub, frac = frac, profile = prior$profile, covariance = prior$cov)
        saveRDS(posterior,file.path(out_dir, "bmind.rds"))
    
        pred <- posterior$A
        colnames(pred) <- celltype
    
        n_gene <- length(dimnames(pred)[[1]])
        donors <- dimnames(pred)[[3]]
    
        ctss <- array(0, dim = c(length(donors),n_gene * dim(pred)[2]))
        rownames(ctss) <- dimnames(pred)[[3]]
        colnames(ctss) <- c(outer(colnames(pred),rownames(pred), FUN = function(a, b) paste0(b, '_', a)))
    
        for(i in seq_along(donors)) {
            ctss[i, ] <- as.vector(t(pred[,,i]))
        }
    
        write.table(ctss,file.path(out_dir, "bmind.tsv"), col.names = TRUE, row.names = TRUE, sep = "\t", quote = FALSE)
    }
    """
    
    # Convert the R function to an R object
    run_bMIND_r = ro.r(r_code)
    
    print("Running bMIND in R")
    # Call the R function with the provided directories
    run_bMIND_r(target_dir, sc_dir, out_dir)