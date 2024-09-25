#!/bin/bash
#SBATCH --job-name="snakemake_%j"
#SBATCH --partition=componc_cpu
#SBATCH --output=logs/snakemake_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

# source /usersoftware/shahs3/users/william1/miniforge3/etc/profile.d/conda.sh
# source /usersoftware/shahs3/users/william1/miniforge3/etc/profile.d/mamba.sh
# module load singularity
# mamba activate snakemake7

# Set default snakemake file if no argument is provided
DEFAULT_SMK_FILE="rules/spectrum.smk"
SMK_FILE="${1:-$DEFAULT_SMK_FILE}"

source /usersoftware/shahs3/users/weinera2/miniconda3/etc/profile.d/conda.sh
conda activate snakemake7
module load singularity

snakemake \
  --use-singularity \
  --use-conda \
  --profile slurm \
  --rerun-triggers mtime \
  --singularity-args "--bind /usersoftware --bind /data1 --bind /home/weinera2" \
  -s "$SMK_FILE" \
  --keep-going \
  --dryrun \