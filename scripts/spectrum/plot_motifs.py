import grafiti as gf
import scanpy as sc
import warnings
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import squidpy as sq
from matplotlib.colors import ListedColormap
from argparse import ArgumentParser
import os
sc.settings.set_figure_params(dpi=80, facecolor='white')
warnings.filterwarnings('ignore')


def get_args():
    p = ArgumentParser()

    p.add_argument('-i', '--input', help='h5ad file containing spectrum data with grafiti embedding + motifs')
    p.add_argument('-ou', '--output_umap', help='UMAP plots of grafiti embedding')
    p.add_argument('-usd', '--umap_sig_density', help='UMAP density plots of mutational signatures')
    p.add_argument('-fw', '--fraction_wgd', help='fraction barplots of grafiti motifs by WGD status')
    p.add_argument('-fs', '--fraction_sig', help='fraction barplots of grafiti motifs by consensus signature')
    p.add_argument('-fc', '--fraction_ct', help='fraction barplots of grafiti motifs by cell type')
    p.add_argument('-fsa', '--fraction_sample', help='fraction barplots of grafiti motifs by sample')
    p.add_argument('-ft', '--fraction_total', help='fraction barplots of grafiti motifs across all samples')
    p.add_argument('-fov', '--fov_annotation', help='FOV annotation plots')
    p.add_argument('-mh', '--motif_heatmap', help='mean expression of input features in each grafiti motif heatmap')
    p.add_argument('-cs', '--centrality_scores', help='centrality scores of grafiti motifs')

    return p.parse_args()



def main():
    argv = get_args()

    adata = sc.read(argv.input)

    # define celltype order
    ct_order = ['CD8+','CD68+','panCK+','Other','Unknown']
    adata.obs['ct_ordered'] = pd.Categorical(values=adata.obs.cell_type, categories=ct_order, ordered=True)
    gm_order = ['GrafitiMotif0','GrafitiMotif1','GrafitiMotif2','GrafitiMotif3','GrafitiMotif4','GrafitiMotif5',
                'GrafitiMotif6','GrafitiMotif7','GrafitiMotif8','GrafitiMotif9','GrafitiMotif10','GrafitiMotif11',
                'GrafitiMotif12','GrafitiMotif13','GrafitiMotif14']
    adata.obs['gm_ordered'] = pd.Categorical(values=adata.obs.grafiti_motif, categories=gm_order, ordered=True)
    wgd_order = ['WGD','Non-WGD']
    adata.obs['wgd_ordered'] = pd.Categorical(values=adata.obs.wgd_status, categories=wgd_order, ordered=True)
    sig_order = ['HRD-Dup','HRD-Del','FBI']
    adata.obs['sig_ordered'] = pd.Categorical(values=adata.obs.consensus_signature, categories=sig_order, ordered=True)
    ct_colors = ['#32A02C','#FF8000','#E31A1C','#999999','#DEDEDE']
    wgd_colors = ['#B42B3E','#3272B5']
    sig_colors = ['#5293C1','#68AC6A','#853213']
    gm_colors = ['#2f4f4f','#8b4513','#6b8e23','#4b0082','#ff0000','#ffa500','#ffff00','#00ff00','#00bfff',
                '#0000ff','#ff00ff','#dda0dd','#ff1493','#7fffd4','#ffdead']
    ct_colors_dict = {'CD8+':'#32A02C','CD68+':'#FF8000','panCK+':'#E31A1C','Other':'#999999','Unknown':'#DEDEDE'}
    gm_colors_dict = {'GrafitiMotif0':'#2f4f4f','GrafitiMotif1':'#8b4513','GrafitiMotif2':'#6b8e23',
                      'GrafitiMotif3':'#4b0082','GrafitiMotif4':'#ff0000',
                      'GrafitiMotif5':'#ffa500',
                      'GrafitiMotif6':'#ffff00','GrafitiMotif7':'#00ff00','GrafitiMotif8':'#00bfff',
                      'GrafitiMotif9':'#0000ff','GrafitiMotif10':'#ff00ff','GrafitiMotif11':'#dda0dd',
                      'GrafitiMotif12':'#ff1493','GrafitiMotif13':'#7fffd4','GrafitiMotif14':'#ffdead'
                    }

    # make umap plots
    fig, ax = plt.subplots(2, 2, figsize=(24, 12), tight_layout=True)
    ax = ax.flatten()
    sc.pl.umap(adata,color="ct_ordered",s=10,add_outline=False,legend_loc='right margin',palette=ct_colors,save=None,ax=ax[0])
    sc.pl.umap(adata,color="wgd_ordered",s=10,add_outline=False,legend_loc='right margin',palette=wgd_colors,save=None,ax=ax[1])
    sc.pl.umap(adata,color="sig_ordered",s=10,add_outline=False,legend_loc='right margin',palette=sig_colors,save=None,ax=ax[2])
    sc.pl.umap(adata,color="grafiti_motif",s=10,add_outline=False,legend_loc='right margin',palette=gm_colors,save=None,ax=ax[3])
    fig.savefig(argv.output_umap, dpi=300, bbox_inches='tight')

    # make fraction barplots
    gf.pl.plot_fraction(adata,"grafiti_motif","wgd_status",color=wgd_colors,save=argv.fraction_wgd)
    gf.pl.plot_fraction(adata,"grafiti_motif","consensus_signature",color=sig_colors,save=argv.fraction_sig)
    gf.pl.plot_fraction(adata,"grafiti_motif","ct_ordered",color=ct_colors,save=argv.fraction_ct)
    gf.pl.plot_fraction(adata,"grafiti_motif","spectrum_sample_id",save=argv.fraction_sample)
    adata.obs['all']='all'
    gf.pl.plot_fraction(adata,"all","grafiti_motif",color=gm_colors,save=argv.fraction_total)


    # plot mutational signature UMAP embedding as a density plot
    xdata = adata.copy()
    sc.pp.subsample(xdata,fraction=0.05)
    sc.tl.embedding_density(xdata, basis='umap', groupby='consensus_signature')
    fig = sc.pl.embedding_density(xdata, basis='umap', key='umap_density_consensus_signature', return_fig=True)
    fig.savefig(argv.umap_sig_density, dpi=300, bbox_inches='tight')

    # find representative WGD and non-WGD FOVs to plot
    ngd_adata = adata[adata.obs.spectrum_sample_id.str.startswith('SPECTRUM-OV-007_S1_RIGHT_FALLOPIAN_TUBE'),:]
    ngd_adata = ngd_adata[ngd_adata.obs.spectrum_fov_id.str.endswith('[46818,10938]')]
    wgd_adata = adata[adata.obs.spectrum_sample_id.str.startswith('SPECTRUM-OV-065'),:]
    wgd_adata = wgd_adata[wgd_adata.obs.spectrum_fov_id.str.endswith('[42822,10882]')]
    fov_ids = [ngd_adata.obs.spectrum_fov_id.values[0],wgd_adata.obs.spectrum_fov_id.values[0]]

    # plot FOVs annotated by panCK signal, cell type, and grafiti motifs
    fig, ax = plt.subplots(len(fov_ids), 3, figsize=(15, len(fov_ids)*5), tight_layout=True)
    for i, fov_id in enumerate(fov_ids):
        temp_adata_image = adata[adata.obs['spectrum_fov_id'] == fov_id]
        sns.scatterplot(x=temp_adata_image.obs['x0'],y=temp_adata_image.obs['y0'],hue=temp_adata_image.X[:, temp_adata_image.var.index.get_loc('panCK')],s=5, ax=ax[i, 0], palette='viridis', rasterized=True)
        ax[i, 0].legend(title='panCK')
        sns.scatterplot(x=temp_adata_image.obs['x0'],y=temp_adata_image.obs['y0'],hue=temp_adata_image.obs['cell_type'],s=5, ax=ax[i, 1], palette=ct_colors_dict, rasterized=True)
        fov_id_suffix = fov_id.split('_')[-1]
        patient_id = fov_id.split('_')[0]
        wgd_status = temp_adata_image.obs.wgd_ordered.unique()[0]
        sig_status = temp_adata_image.obs.sig_ordered.unique()[0]
        ax[i, 1].set_title(f'{patient_id}, {wgd_status}, {sig_status}, FOV:{fov_id_suffix}')
        sns.scatterplot(x=temp_adata_image.obs['x0'],y=temp_adata_image.obs['y0'],hue=temp_adata_image.obs['grafiti_motif'],s=5, ax=ax[i, 2], palette=gm_colors_dict, rasterized=True)
        ax[i, 2].legend(bbox_to_anchor=(1.5,1), loc='upper right')
    fig.savefig(argv.fov_annotation, dpi=300, bbox_inches='tight')

    # plot a heatmap of mean expression of input features in each grafiti motif
    fig = sc.pl.matrixplot(adata,adata.var.index.tolist(),groupby='grafiti_motif',standard_scale="var",
                     return_fig=True)
    fig.savefig(argv.motif_heatmap, dpi=300, bbox_inches='tight')

    # The scores calculated are closeness centrality, degree centrality and clustering coefficient with the following properties:
    # (1) closeness centrality - measure of how close the group is to other nodes.
    # (2) clustering coefficient - measure of the degree to which nodes cluster together.
    # (3) degree centrality - fraction of non-group members connected to group members.
    sq.gr.centrality_scores(adata, cluster_key="grafiti_motif")
    gm_cmap = ListedColormap(gm_colors)
    sq.pl.centrality_scores(adata, cluster_key="grafiti_motif", figsize=(20, 5), s=380, palette=gm_cmap, save=os.getcwd()+'/'+argv.centrality_scores)
    plt.show()



if __name__ == '__main__':
    main()