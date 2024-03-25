import scanpy as sc
import os

dfolder = "/data1/shahs3/users/pourmalm/for_grafiti/"

def get_adata(adata):
    cluster_copy = os.path.join(dfolder,adata)
    if os.path.exists(cluster_copy):
        return sc.read(cluster_copy)
    elif os.path.exists(adata):
        return sc.read(adata)
    else:
        raise OSError("Can't find {}".format(adata))

def spectrum():
    adata = "SPECTRUM_squidpy.h5ad"
    return get_adata(adata)

def merck():
    adata = "MERCK_squidpy.h5ad"
    return get_adata(adata)

def melanoma():
    adata = "MELANOMA_allUT_squidpy.h5ad"
    return get_adata(adata)

def bodenmiller():
    adata = "bodenmiller_squidpy.h5ad"
    return get_adata(adata)

def list_datasets():
    return ["spectrum", "melanoma","bodenmiller","merck"]