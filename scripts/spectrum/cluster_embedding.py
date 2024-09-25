import grafiti as gf
import scanpy as sc
import warnings
from argparse import ArgumentParser
sc.settings.set_figure_params(dpi=80, facecolor='white')
warnings.filterwarnings('ignore')


def get_args():
    p = ArgumentParser()

    p.add_argument('-i', '--input', help='h5ad file containing spectrum data with grafiti embedding')
    p.add_argument('-r', '--resolution', type=float, help='resolution parameter for clustering')
    p.add_argument('-m', '--method', help='clustering method')
    p.add_argument('-nn', '--n_neighbors', type=int, help='number of neighbors for building graph')
    p.add_argument('-me', '--metric', help='metric for building graph')
    p.add_argument('-k', '--k', type=int, help='number of clusters')
    p.add_argument('-mi', '--max_iter_clust', type=int, help='max number of iterations for clustering')
    p.add_argument('-cn', '--compute_neighbors', type=bool, help='compute neighbors for clustering')
    p.add_argument('-o', '--output', help='output h5ad file for storing the clustered embedding')

    return p.parse_args()



def main():
    argv = get_args()

    print('Arguments:', argv)

    print('Reading the adata object...')

    adata = sc.read(argv.input)

    print('Performing clustering...')

    # perform clustering
    gf.tl.find_motifs(adata, resolution=argv.resolution, method=argv.method, n_neighbors=argv.n_neighbors, metric=argv.metric, k=argv.k, max_iter=argv.max_iter_clust, 
                  compute_neighbors=argv.compute_neighbors)
    
    print('Writing the adata object with clusters added...')

    # save the adata object as h5ad
    adata.write(argv.output)

    print('Done writing file:', argv.output)


if __name__ == '__main__':
    main()
    print('done with main()')
