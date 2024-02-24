import squidpy as sq
import scanpy as sc
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib

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