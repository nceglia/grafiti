import scanpy as sc
import os

dfolder = "/work/shah/users/pourmalm/mpif_data/for_sappy/"
lfolder = "../notebooks/datasets/"
# if not os.path.exists(dfolder) and not os.path.exists(lfolder):
#     os.makedirs(lfolder)

def get_adata(adata):
    local_copy = os.path.join(lfolder,adata)
    juno_copy = os.path.join(dfolder,adata)
    if os.path.exists(juno_copy):
        return sc.read(juno_copy)
    elif os.path.exists(local_copy):
        return sc.read(local_copy)
    elif os.path.exists("/work/shah/ceglian/SPECTRUM_squidpy.h5ad"):
        return sc.read("/work/shah/ceglian/SPECTRUM_squidpy.h5ad")
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