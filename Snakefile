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
        )