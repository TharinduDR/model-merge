#!/bin/bash
#SBATCH -p parallel
#SBATCH --array=0-0
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=log/prep_%A_%a.log

# --- do all environment setup FIRST, while nounset is OFF ---
source /etc/profile
module add anaconda3/2023.09
source activate /storage/hpc/37/ranasint/conda_envs/llm_exp

# --- now turn on strict mode for our own commands ---
set -euo pipefail

SCRATCH=/scratch/hpc/37/ranasint
export HF_HOME="${SCRATCH}/hf_cache"
export HF_TOKEN=
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"; mkdir -p log

DATA_RAW="${SCRATCH}/data_raw"
DATA="${SCRATCH}/data"
TOK="${SCRATCH}/tokenizers"

langs=(sinhala)
lang="${langs[$SLURM_ARRAY_TASK_ID]}"

if [ ! -f "${DATA}/${lang}/_SUCCESS" ]; then
    python -u src/data_pipeline.py --language "$lang" \
        --data-raw "${DATA_RAW}" --out "${DATA}/${lang}" --near-dedup
fi
[ -f "${DATA}/${lang}/_SUCCESS" ] || { echo "data build failed for ${lang}; not training tokenizer"; exit 1; }

if [ ! -f "${TOK}/${lang}.model" ]; then
    python -u src/train_lang_tokenizer.py --data "${DATA}/${lang}" --out "${TOK}/${lang}"
fi
echo "prep done for ${lang}"