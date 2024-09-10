import grafiti as gf
import scanpy as sc
import warnings
import matplotlib.pyplot as plt
import umap as umap_ext
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import GaussianMixture
from argparse import ArgumentParser
sc.settings.set_figure_params(dpi=80, facecolor='white')
# warnings.filterwarnings('ignore')
import os


def get_args():
    p = ArgumentParser()

    p.add_argument('-i', '--input', help='h5ad file containing spectrum data')
    # GAE arguments
    p.add_argument('-l', '--layers', type=int, nargs='+', help='list of layers for GAE. Each element is the size of each layer.')
    p.add_argument('-lr', '--learning_rate', type=float, help='learning rate for training')
    p.add_argument('-g', '--gamma', type=float, help='gamma parameter for learning rate scheduler')
    p.add_argument('-mi', '--max_iter', type=int, help='max number of epochs for training')
    p.add_argument('-dt', '--distance_thresh', type=float, help='distance threshold between cells for building graph')
    p.add_argument('-e', '--exponent', type=float, help='exponent for distance scaling')
    p.add_argument('-ds', '--distance_scale', type=float, help='distance scaling factor')
    p.add_argument('-a', '--alpha', type=float, help='alpha parameter for loss function')
    p.add_argument('-t', '--threshold', type=float, help='threshold for stopping training')
    p.add_argument('-s', '--seed', type=int, help='random seed for training')
    # clustering arguments
    p.add_argument('-r', '--resolution', type=float, help='resolution parameter for clustering')
    p.add_argument('-m', '--method', type=str, help='clustering method')
    p.add_argument('-nn', '--n_neighbors', type=int, help='number of neighbors for building graph')
    p.add_argument('-me', '--metric', type=str, help='metric for building graph')
    p.add_argument('-k', '--k', type=int, help='number of clusters')
    p.add_argument('-mic', '--max_iter_clust', type=int, help='max number of iterations for clustering')
    # output arguments
    p.add_argument('-mp', '--model_path', help='output file for storing the trained model')
    p.add_argument('-o', '--output', help='output h5ad file for storing the grafiti embedding')
    p.add_argument('-lc', '--loss_curve', help='output file for storing the loss curve during training')

    return p.parse_args()


def plot_loss_curve(gae, argv):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True)
    ax.plot(range(len(gae.losses)), gae.losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('layers: {}, lr: {}, gamma: {}, max_iter: {}, alpha: {}\ndistance_thresh: {}, exponent: {}, distance_scale: {}, seed: {}, loss_thresh: {}'.format(argv.layers, argv.learning_rate, argv.gamma, argv.max_iter, argv.alpha, argv.distance_thresh, argv.exponent, argv.distance_scale, argv.seed, argv.threshold))
    fig.savefig(argv.loss_curve)


def main():
    argv = get_args()

    adata = sc.read(argv.input)

    # remove extraneous information
    columns_to_drop = ['grafiti_motif','tme_inclusion_status','impact_signature','BRCA_gene_mutation_status','all_total','all_arm','all_chrom',
                    'all_total_gain','all_arm_gain','all_chrom_gain','all_total_loss','all_arm_loss','all_chrom_loss','internal_total',
                    'internal_arm','internal_chrom','internal_total_gain','internal_arm_gain','internal_chrom_gain','internal_total_loss',
                    'internal_arm_loss','internal_chrom_loss','terminal_total','terminal_arm','terminal_chrom','terminal_total_gain',
                    'terminal_arm_gain','terminal_chrom_gain','terminal_total_loss','terminal_arm_loss','terminal_chrom_loss','n_cells',
                    'all_total_rate','all_arm_rate','all_chrom_rate','all_total_gain_rate','all_arm_gain_rate','all_chrom_gain_rate',
                    'all_total_loss_rate','all_arm_loss_rate','all_chrom_loss_rate','internal_total_rate','internal_arm_rate',
                    'internal_chrom_rate','internal_total_gain_rate','internal_arm_gain_rate','internal_chrom_gain_rate','terminal_arm_rate',
                    'terminal_chrom_rate','terminal_total_gain_rate','terminal_arm_gain_rate','terminal_chrom_gain_rate',
                    'terminal_total_loss_rate','terminal_arm_loss_rate','terminal_chrom_loss_rate','terminal_wgd_rate','internal_wgd_rate',
                    'total_num_cells']
    adata.obs.drop(columns=columns_to_drop, inplace=True)
    keys = ['X_grafiti', 'X_grafiti_prob', 'X_umap']
    for key in keys:
        if key in adata.obsm:
            del adata.obsm[key]

    # create model with data
    gae = gf.ml.GAE(adata, layers=argv.layers, distance_threshold=argv.distance_thresh, exponent=argv.exponent, distance_scale=argv.distance_scale, lr=argv.learning_rate, gamma=argv.gamma, alpha=argv.alpha, seed=argv.seed)

    # train model
    gae.train(argv.max_iter, update_interval=10, threshold=argv.threshold)

    # save model
    gae.save(argv.model_path)

    # add the grafiti embedding to adata
    gae.load_embedding(adata, encoding_key="X_grafiti")

    # plot the loss curve
    plot_loss_curve(gae, argv)

    # # compute UMAP on grafiti embedding
    # gf.tl.umap(adata, encoding_key="X_grafiti", n_neighbors=30, max_iter=100, min_dist=0.05, spread=4, metric="euclidean", 
    #        scanpy=False, neighbors_key="grafiti_neighbors")

    xdata = adata.copy()
    sc.pp.subsample(xdata,fraction=0.2)
    cluster_key='grafiti_motif'
    prefix="GrafitiMotif"
    embedding_key="X_grafiti",
    X=normalize(xdata.obsm["X_grafiti"])
    gm = GaussianMixture(n_components=argv.k, random_state=0, max_iter=400,verbose=True,covariance_type="spherical").fit(X)
    X=normalize(adata.obsm["X_grafiti"])
    adata.obs["grafiti_motif"] = ["{}{}".format(prefix,x) for x in gm.predict(X).tolist()]
    adata.uns["grafiti_motif_proba"] = gm.predict_proba(X)
    for gm, prob in zip(sorted(set(adata.obs[cluster_key])),adata.uns["{}_proba".format(cluster_key)].T):
        adata.obs["{}_proba".format(gm)] = prob

    ldm = umap_ext.UMAP(n_epochs=100,
                    n_neighbors=30,
                    min_dist=0.05,
                    spread=4,
                    metric="euclidean")
    embd = ldm.fit_transform(X)
    adata.obsm["X_umap"] = embd

    print(adata)

    # # before the clustering step
    # # print the full path of the output file that is being written
    # print('before clustering')
    # print('writing h5ad file to: {}'.format(argv.output))
    # print('current working directory', os.getcwd())
    
    # # perform clustering
    # gf.tl.find_motifs(adata, resolution=argv.resolution, method=argv.method, n_neighbors=argv.n_neighbors, metric=argv.metric, k=argv.k, max_iter=argv.max_iter_clust, 
    #               compute_neighbors=False)
    # print('done with clustering')

    
    # # print the full path of the output file that is being written
    # print('after clustering')
    # print('writing h5ad file to: {}'.format(argv.output))
    # print('current working directory', os.getcwd())

    adata.write(argv.output)

    print(adata)


if __name__ == '__main__':
    main()
    print('done with main()')
