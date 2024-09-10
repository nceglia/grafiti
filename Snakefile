import numpy as np
import pandas as pd

np.random.seed(2794834348)

configfile: "config.yaml"

bad_runs = []

rule all:
    input:
        expand(
            'plots/spectrum/{run}/umap_plots.pdf',
            run=[
                # 'R1.0'
                d for d in config['spectrum_params']
                if (d not in bad_runs)
            ]
        ),
        expand(
            'plots/spectrum/{run}/fov_plots.pdf',
            run=[
                # 'R1.0'
                d for d in config['spectrum_params']
                if (d not in bad_runs)
            ]
        )


rule run_grafiti:
    input: '/data1/shahs3/users/pourmalm/for_grafiti/spectrum_all.h5ad'
    params:
        layers = lambda wildcards: config['spectrum_params'][wildcards.run]['layers'],
        learning_rate = lambda wildcards: config['spectrum_params'][wildcards.run]['learning_rate'],
        gamma = lambda wildcards: config['spectrum_params'][wildcards.run]['gamma'],
        max_iter = lambda wildcards: config['spectrum_params'][wildcards.run]['max_iter'],
        distance_thresh = lambda wildcards: config['spectrum_params'][wildcards.run]['distance_thresh'],
        exponent = lambda wildcards: config['spectrum_params'][wildcards.run]['exponent'],
        distance_scale = lambda wildcards: config['spectrum_params'][wildcards.run]['distance_scale'],
        alpha = lambda wildcards: config['spectrum_params'][wildcards.run]['alpha'],
        threshold = lambda wildcards: config['spectrum_params'][wildcards.run]['threshold'],
        seed = lambda wildcards: config['spectrum_params'][wildcards.run]['seed'],
        resolution = lambda wildcards: config['spectrum_params'][wildcards.run]['resolution'],
        method = lambda wildcards: config['spectrum_params'][wildcards.run]['method'],
        n_neighbors = lambda wildcards: config['spectrum_params'][wildcards.run]['n_neighbors'],
        metric = lambda wildcards: config['spectrum_params'][wildcards.run]['metric'],
        k = lambda wildcards: config['spectrum_params'][wildcards.run]['k'],
        max_iter_clust = lambda wildcards: config['spectrum_params'][wildcards.run]['max_iter_clust'],
    output:
        model_path = 'analysis/spectrum/{run}/240628_spectrum_model',
        h5ad = 'analysis/spectrum/{run}/240628_spectrum.h5ad',
        loss_curve = 'plots/spectrum/{run}/loss_curve.png'
    log: 'logs/spectrum/{run}/run_grafiti.log'
    resources: 
        mem_mb=1000 * 96,
        time = "06:00:00", 
        partition = 'componc_cpu'
    shell:
        'source /usersoftware/shahs3/users/weinera2/miniconda3/bin/activate /usersoftware/shahs3/users/weinera2/miniconda3/envs/grafiti2 ; '
        'python3 scripts/spectrum/run_grafiti.py '
        '-i {input} '
        '-l {params.layers} '
        '-lr {params.learning_rate} '
        '-g {params.gamma} '
        '-mi {params.max_iter} '
        '-dt {params.distance_thresh} '
        '-e {params.exponent} '
        '-ds {params.distance_scale} '
        '-a {params.alpha} '
        '-t {params.threshold} '
        '-s {params.seed} '
        '-r {params.resolution} '
        '-m {params.method} '
        '-nn {params.n_neighbors} '
        '-me {params.metric} '
        '-k {params.k} '
        '-mic {params.max_iter_clust} '
        '-mp {output.model_path} '
        '-o {output.h5ad} '
        '-lc {output.loss_curve} &> {log}'


rule plot_motifs:
    input: 'analysis/spectrum/{run}/240628_spectrum.h5ad'
    output: 
        umaps = 'plots/spectrum/{run}/umap_plots.pdf',
        umap_sig_density = 'plots/spectrum/{run}/umap_sig_density.pdf',
        fraction_wgd = 'plots/spectrum/{run}/fraction_wgd.pdf',
        fraction_sig = 'plots/spectrum/{run}/fraction_sig.pdf',
        fraction_ct = 'plots/spectrum/{run}/fraction_ct.pdf',
        fraction_sample = 'plots/spectrum/{run}/fraction_sample.pdf',
        fraction_total = 'plots/spectrum/{run}/fraction_plots.pdf',
        fov_plots = 'plots/spectrum/{run}/fov_plots.pdf'
    log: 'logs/spectrum/{run}/plot_motifs.log'
    resources: 
        mem_mb=1000 * 64,
        time = "06:00:00",
        partition = 'componc_cpu'
    shell:
        'source /usersoftware/shahs3/users/weinera2/miniconda3/bin/activate /usersoftware/shahs3/users/weinera2/miniconda3/envs/grafiti2 ; '
        'python3 scripts/spectrum/plot_motifs.py '
        '-i {input} '
        '-ou {output.umaps} '
        '-usd {output.umap_sig_density} '
        '-fw {output.fraction_wgd} '
        '-fs {output.fraction_sig} '
        '-fc {output.fraction_ct} '
        '-fsa {output.fraction_sample} '
        '-ft {output.fraction_total} '
        '-fov {output.fov_plots} &> {log}'
