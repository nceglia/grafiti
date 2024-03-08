import squidpy as sq
import scanpy as sc
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib
import pandas as pd
import numpy as np
from statannotations.Annotator import Annotator
import itertools
from scipy import sparse
import collections
import operator
from sklearn import preprocessing

from ..tools._tools import get_fov_graph

grafiti_colors = [
    "#ff0000",  # Bright Red
    "#00ff00",  # Bright Green
    "#0000ff",  # Bright Blue
    "#ffff00",  # Bright Yellow
    "#ff00ff",  # Bright Pink
    "#00ffff",  # Bright Cyan
    "#ff6600",  # Bright Orange
    "#9900ff",  # Bright Purple
    "#00ff99",  # Bright Aqua
    "#ff3399",  # Hot Pink
    "#3366ff",  # Sky Blue
    "#33cc33",  # Lime Green
    "#cc33ff",  # Neon Purple
    "#ffcc00",  # Golden Yellow
    "#ff99cc",  # Pastel Pink
    "#66ffcc",  # Turquoise
    "#cc9966",  # Bronze
    "#9999ff",  # Lavender
    "#ccff66",  # Neon Green
    "#ff6666",  # Salmon Red
    "#66ccff",  # Light Blue
    "#ccffcc",  # Mint Green
    "#ffccff",  # Pale Pink
    "#ccccff",  # Periwinkle
    "#ff9966",  # Coral
    "#66cccc",  # Aqua Green
    "#ff6666",  # Soft Red
    "#cc6699",  # Mauve
    "#9966cc",  # Indigo
    "#669966",  # Olive Green
]

def fovs(adata, cluster_key="sap", fov_key="sample_fov"):
    sq.pl.spatial_scatter(adata, color=cluster_key, shape=None, library_key=fov_key)

def plot_fraction(adata,category,variable,save=None,color=grafiti_colors, figsize=(10,6)):
    df = adata.obs
    count_df = df.groupby([category, variable]).size().unstack(fill_value=0)
    proportion_df = count_df.divide(count_df.sum(axis=1), axis=0)
    proportion_df.plot(kind="bar", stacked=True, figsize=figsize,color=color)
    plt.ylabel("Fraction")
    plt.ylim([0, 1])
    plt.legend(title=variable, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save != None:
        plt.savefig(save)

def umap(adata, key="grafiti", save=None, add_outline=False, s=20, figsize=(9,6)):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    sc.pl.umap(adata,color=key,ax=ax,show=False,add_outline=add_outline,s=s)
    fig.tight_layout()
    if save != None:
        fig.savefig(save)
def plot_fov_graph(adata, fov_id, use_coords=True, n_cols=4, s=10, save=None, cluster_key="grafiti_motif", spatial_key="spatial", fov_key="sample_fov", alpha=0.8, width=0.5, figsize=(6,6),bbox_to_anchor=(1.2,0.9)):
    if type(fov_id) == str:
        fov_id = [fov_id]
    if len(fov_id) < n_cols:
        n_rows = 1
        n_cols = len(fov_id)
    else:
        n_rows = int(len(fov_id) / n_cols)
    sub = adata[adata.obs[fov_key].isin(fov_id)]
    sap_order = []
    order_to_sap = dict()
    for x in set(adata.obs[cluster_key]):
        i = int(x.replace("GrafitiMotif",""))
        sap_order.append(i)
        order_to_sap[i] = x
    sap_order = list(sorted(sap_order))
    sap_sorted = [order_to_sap[i] for i in sap_order]

    cmap = dict(zip(sap_sorted,grafiti_colors))
    for x in list(cmap.keys()):
        if x not in sub.obs[cluster_key].tolist():
            del cmap[x]
    handles = []
    for community_id, color in cmap.items():
        handle = mpatches.Patch(color=color, label=f'{community_id}')
        handles.append(handle)
    fig,ax=plt.subplots(n_rows,n_cols,figsize=figsize)
    cur_id = 0
    for cur_row in range(n_rows):
        for cur_col in range(n_cols):
            subx = sub[sub.obs[fov_key] == fov_id[cur_id]]
            if type(ax) != matplotlib.axes._axes.Axes:
                if len(ax.shape) == 1:
                    axcur = ax[cur_col]
                else:
                    axcur = ax[cur_row][cur_col]
            else:
                axcur = ax
            G = get_fov_graph(sub, fov_id[cur_id], fov_key=fov_key)
            node_colors = []
            for n in subx.obs[cluster_key]:
                node_colors.append(cmap[n])
            if use_coords:
                pos=subx.obsm[spatial_key]
            else:
                pos=nx.spring_layout(G)
            nx.draw_networkx(G, pos=pos,ax=axcur,node_size=s,with_labels=False,alpha=alpha,width=width,node_color=node_colors)
            axcur.set_title(fov_id[cur_id])
            cur_id += 1
    fig.legend(handles=handles, loc='upper right',bbox_to_anchor=bbox_to_anchor)
    fig.tight_layout()
    if save != None:
        fig.savefig(save)

def crosstab_heatmap(adata, variable1, variable2, figsize=(6,6), save=None):
    fig,ax=plt.subplots(1,1,figsize=figsize)
    df = adata.obs[[variable1,variable2]]
    df=pd.crosstab(df[variable1],df[variable2],normalize='index')
    ax = sns.heatmap(df,ax=ax)
    fig.tight_layout()
    if save:
        fig.savefig

def fov(adata,fov):
    adata.obs["X"] = adata.obsm["spatial"].T[0]
    adata.obs["Y"] = adata.obsm["spatial"].T[1]
    sc.pl.embedding(adata,basis="spatial",color="grafiti",s=100, alpha=0.5)

def set_colors(adata, columns, color_list=None):
    i = 0
    main_color_map = dict()
    adata = adata.copy()
    if color_list == None:
        colors = grafiti_colors.copy() + grafiti_colors.copy() + grafiti_colors.copy()
    for x in columns:
        ct = []
        for i, val in enumerate(set(adata.obs[x].tolist())):
            c = colors.pop(i)
            ct.append(c)
            main_color_map[val] = c
        adata.uns["{}_colors".format(x)] = ct
    return adata

def motif_by_feature(adata,feature,fov_id,fov_key="sample_fov",coord_key="spatial",sap_key="grafiti",figsize=(18,4),s=30,add_outline=False,color="celltype",ct_color=grafiti_colors):
    adata = adata[adata.obs[fov_key] == fov_id]

    sap_order = []
    order_to_sap = dict()
    for x in set(adata.obs["grafiti"]):
        i = int(x.replace("GrafitiMotif",""))
        sap_order.append(i)
        order_to_sap[i] = x
    sap_order = list(sorted(sap_order))
    sap_sorted = [order_to_sap[i] for i in sap_order]
    cmap = dict(zip(sap_sorted,grafiti_colors))
    
    fig, ax = plt.subplots(1,4,figsize=figsize)
    sc.pl.embedding(adata,basis=coord_key,color=[feature],title=[feature],s=s, add_outline=add_outline,ax=ax[0],show=False)
    sc.pl.embedding(adata,basis=coord_key,color="grafiti",title=["Grafiti Spatial Motifs"],s=s, add_outline=add_outline,ax=ax[2],show=False,palette=cmap)
    sc.pl.embedding(adata,basis=coord_key,color=color,title=["Grafiti Spatial Motifs"],s=s, add_outline=add_outline,ax=ax[3],show=False,palette=ct_color)
    if feature in adata.var.index.tolist():
        f = adata.X[:,adata.var.index.tolist().index(feature)].tolist()
    elif feature in adata.obs.columns:
        f = adata.obs[feature]
    else:
        raise ValueError("Feature {} not found in obs or var.".format(feature))
    df = adata.obs.copy()
    df[feature] = f
    sns.boxplot(data=df,x=sap_key,y=feature,ax=ax[1],palette=cmap)
    fig.tight_layout()

def boxplot(adata, groupby, variable, splitby, summarizeby, box=True, split_order=None, order=None, swarm=True, point_outline=1, alpha=0.9,point_size=6, figsize=(7,5), bbox=(1.03, 1), dpi=300, save=None):
    def get_expression(gene):
        if type(adata.X) != sparse.csr_matrix:
            adata.X = sparse.csr_matrix(adata.X)
        return adata.X[:,adata.var.index.tolist().index(gene)].T.todense().tolist()[0]
    df = adata.obs
    if variable not in df.columns:
        df[variable] = get_expression(variable)
    splts = []
    grps = []
    summarize = []
    values = []
    for grp in set(adata.obs[groupby]):
        gdf = df[df[groupby] == grp]
        for s in set(gdf[splitby]):
            sdf = gdf[gdf[splitby] == s]
            for smz in set(sdf[summarizeby]):
                smzdata = sdf[sdf[summarizeby] == smz]
                summarize.append(smz)
                grps.append(grp)
                splts.append(s)
                values.append(np.mean(smzdata[variable]))
    df = pd.DataFrame.from_dict({summarizeby:summarize,groupby:grps,splitby:splts,variable:values})
    if split_order == None:
        split_order = list(set(df[splitby]))
    if order == None:
        order = list(set(df[groupby])) 
    split_combos = list(itertools.combinations(split_order,2))
    pairs = []
    for o in order:
        for c in split_combos:
            pairs.append([(o,c[0]),(o,c[1])])
    fig,ax = plt.subplots(1,1,figsize=figsize)
    if not box:
        sns.violinplot(data=df,x=groupby, y=variable, hue=splitby, ax=ax, order=order, hue_order=split_order)
    else:
        sns.boxplot(data=df,x=groupby, y=variable, hue=splitby, ax=ax, order=order, hue_order=split_order)
    if swarm:
        sns.swarmplot(data=df,x=groupby, y=variable, hue=splitby,ax=ax,order=order,hue_order=split_order, dodge=True, s=point_size,linewidth=point_outline, alpha=alpha)
    annot = Annotator(ax, pairs, data=df, x=groupby, y=variable, order=order, hue=splitby, hue_order=split_order)
    annot.configure(test='Mann-Whitney', verbose=2)
    annot.apply_test()
    annot.annotate()
    plt.legend(title=variable, loc="upper left", bbox_to_anchor=bbox)
    plt.xticks(rotation=90)
    if save != None:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')

def plot_logo(cell_counts, cell_distances, title="",save=None,spring_scale=0.4,fontsize=12, figsize=(4,4),palette=None,alpha=1.):
    G = nx.Graph()
    node_colors = []
    for cell_type, count in cell_counts.items():
        G.add_node(cell_type, size=count)
        if palette != None:
            node_colors.append(palette[cell_type])
    for cell1, cell2, distance in cell_distances:
        G.add_edge(cell1, cell2, weight=distance)
    fig,ax = plt.subplots(1,1,figsize=figsize)
    pos = nx.spring_layout(G,scale=spring_scale,weight="weight")
    sizes = [G.nodes[node]['size'] for node in G.nodes]  # scale size for visibility
    sizes = preprocessing.MinMaxScaler(feature_range=(100,2000)).fit_transform(np.array(sizes).reshape(-1,1)).reshape(1,-1).tolist()[0]
    nx.draw_networkx_nodes(G, pos, node_size=sizes,ax=ax,node_color=node_colors,alpha=alpha)
    edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = preprocessing.MinMaxScaler(feature_range=(0,5)).fit_transform(np.array(edge_widths).reshape(-1,1)).reshape(1,-1).tolist()[0]
    nx.draw_networkx_edges(G, pos, width=edge_widths,ax=ax,alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=fontsize, font_family='helvetica',ax=ax, font_weight="bold",alpha=0.8)
    ax.set_xlim([1.2*x for x in ax.get_xlim()])
    ax.set_ylim([1.2*y for y in ax.get_ylim()])
    #fig.tight_layout()
    plt.axis('off')

    plt.title(title)
    if save:
        plt.savefig(save)

def generate_motif_logo(adata, motif, cluster_key="grafiti_motif", normalize_by_fov=False, fov_key="sample_fov", run_de=False, fontsize=12, phenotype_key="cell_type", figsize=(3,3), spring_scale=0.4, exclude_phenotypes=[], save=None, title="", min_percentage=0.1,palette=None,alpha=1.):
    adata = adata[adata.obs[cluster_key] == motif] 
    tcounts = collections.defaultdict(int)
    num_images = len(set(adata.obs[fov_key]))
    if normalize_by_fov:
        for x in set(adata.obs[fov_key]):
            fcounts = collections.defaultdict(int)
            df = adata[adata.obs[fov_key] == x].obs
            if len(df.index) > 0:
                total = len(df.index)
                counts = df[phenotype_key].tolist()
                for c in counts:
                    fcounts[c] += 1
                for ct, ctnum in fcounts.items():
                    perc = ctnum / total
                    if perc > min_percentage:
                        tcounts[ct] += perc 
    else:
        fcounts = collections.defaultdict(int)
        df = adata.obs
        if len(df.index) > 0:
            total = len(df.index)
            counts = df[phenotype_key].tolist()
            for c in counts:
                fcounts[c] += 1
            for ct, ctnum in fcounts.items():
                perc = ctnum / total
                if perc > min_percentage:
                    tcounts[ct] += perc 
    sorted_counts = list(reversed(sorted(tcounts.items(), key=operator.itemgetter(1))))
    tcounts = dict()
    for ct, counts in sorted_counts:
        if ct not in exclude_phenotypes:
            tcounts[ct] = counts / num_images
    pairs = list(itertools.combinations(list(tcounts.keys()),2))
    pair_distances = []
    for p in pairs:
        cdata =adata[adata.obs[phenotype_key].isin(p)]
        mask = (cdata.obsp["spatial_connectivities"] == 1)
        distances = cdata.obsp["spatial_distances"]
        distances = distances.multiply(mask)
        pair_distances.append((p[0],p[1],np.median(distances.data)))
    for p in list(tcounts.keys()):
        cdata =adata[adata.obs[phenotype_key].isin([p])]
        mask = (cdata.obsp["spatial_connectivities"] == 1)
        distances = cdata.obsp["spatial_distances"]
        distances = distances.multiply(mask)
        mdist = np.median(distances.data)
        pair_distances.append((p,p,mdist))
    if palette == None:
        palette= dict(zip(list(tcounts.keys()), grafiti_colors))
    plot_logo(tcounts, pair_distances, title=title, save=save, spring_scale=spring_scale, figsize=figsize, fontsize=fontsize, palette=palette,alpha=alpha)

def split_boxplot(adata, groupby, variable, summarizeby, box=True, order=None, swarm=True, point_outline=1, alpha=0.9,point_size=6, figsize=(7,5), bbox=(1.03, 1), dpi=300, save=None):
    def get_expression(gene):
        if type(adata.X) != sparse.csr_matrix:
            adata.X = sparse.csr_matrix(adata.X)
        return adata.X[:,adata.var.index.tolist().index(gene)].T.todense().tolist()[0]
    df = adata.obs
    if variable not in df.columns:
        df[variable] = get_expression(variable)
    splts = []
    grps = []
    summarize = []
    values = []
    for grp in set(adata.obs[groupby]):
        sdf = df[df[groupby] == grp]
        for smz in set(sdf[summarizeby]):
            smzdata = sdf[sdf[summarizeby] == smz]
            summarize.append(smz)
            grps.append(grp)
            values.append(np.mean(smzdata[variable]))
    df = pd.DataFrame.from_dict({summarizeby:summarize,groupby:grps,variable:values})
    if order == None:
        order = list(set(df[groupby])) 
    pairs = list(itertools.combinations(order,2))
    fig,ax = plt.subplots(1,1,figsize=figsize)
    if not box:
        sns.violinplot(data=df,x=groupby, y=variable, ax=ax, order=order)
    else:
        sns.boxplot(data=df,x=groupby, y=variable, ax=ax, order=order)
    if swarm:
        sns.swarmplot(data=df,x=groupby, y=variable,ax=ax,order=order, dodge=True, s=point_size,linewidth=point_outline, alpha=alpha)
    annot = Annotator(ax, pairs, data=df, x=groupby, y=variable, order=order)
    annot.configure(test='Mann-Whitney', verbose=2)
    annot.apply_test()
    ax, test_results= annot.annotate()
    # fig.legend(loc='upper left', bbox_to_anchor=bbox)
    plt.legend(title=variable, loc="upper left", bbox_to_anchor=bbox)
    plt.xticks(rotation=90)
    if save != None:
        fig.savefig(save, dpi=dpi, bbox_inches='tight')
    return list(zip(pairs,[x.formatted_output for x in test_results]))
