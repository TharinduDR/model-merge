#!/bin/bash
# Per-language corpus MERGE + tokenizer training. Fully OFFLINE: reads pre-downloaded
# files (HPLT from data_raw, CulturaX/MADLAD from the HF cache). Do the actual dataset
# DOWNLOADS on an internet-capable node first. CPU + big scratch, NOT a GPU job.
#SBATCH -p parallel              # <-- set to your real CPU queue
#SBATCH --array=0-0              # one task per language; widen if you add languages
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=log/prep_%A_%a.log
set -euo pipefail

source /etc/profile
module add anaconda3/2023.09
source activate /storage/hpc/37/ranasint/conda_envs/llm_exp

SCRATCH=/scratch/hpc/37/ranasint
export HF_HOME="${SCRATCH}/hf_cache"       # where CulturaX/MADLAD were cached
export HF_TOKEN=
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"; mkdir -p log

DATA_RAW="${SCRATCH}/data_raw"             # HPLT lives here (hplt_<code>/)
DATA="${SCRATCH}/data"                     # merged corpus output (scratch)
TOK="${SCRATCH}/tokenizers"                # tokenizers output (scratch)

langs=(sinhala)
lang="${langs[$SLURM_ARRAY_TASK_ID]}"

# 1) Merge + dedup from local files. Guard on the _SUCCESS sentinel, not the dir.
if [ ! -f "${DATA}/${lang}/_SUCCESS" ]; then
    python -u src/data_pipeline.py --language "$lang" \
        --data-raw "${DATA_RAW}" --out "${DATA}/${lang}" --near-dedup
fi
[ -f "${DATA}/${lang}/_SUCCESS" ] || { echo "data build failed for ${lang}; not training tokenizer"; exit 1; }

# 2) Train the language tokenizer only if the corpus build succeeded.
if [ ! -f "${TOK}/${lang}.model" ]; then
    python -u src/train_lang_tokenizer.py --data "${DATA}/${lang}" --out "${TOK}/${lang}"
fi
echo "prep done for ${lang}"
