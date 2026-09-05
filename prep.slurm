#!/bin/bash
# Per-language corpus build + tokenizer training. This is CPU + network + big scratch,
# NOT a GPU job, so keep it OFF the astro/GPU nodes. Also: HEC compute nodes may lack
# outbound internet -- if so, run this stage on a login/data-mover node or wherever the
# datasets can actually be downloaded, and land data/ + tokenizers/ on shared scratch.
#SBATCH -p parallel              # <-- adjust to your CPU queue; GPU docs don't name it
#SBATCH --array=0-0            # one task per language; widen if you add languages
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=log/prep_%A_%a.log


source /etc/profile
module add anaconda3/2023.09
source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"; mkdir -p log

langs=(sinhala)
lang="${langs[$SLURM_ARRAY_TASK_ID]}"

if [ ! -d "data/${lang}" ]; then
    python src/data_pipeline.py --language "$lang" --out "data/${lang}" --near-dedup
fi
if [ ! -f "tokenizers/${lang}.model" ]; then
    python src/train_lang_tokenizer.py --data "data/${lang}" --out "tokenizers/${lang}"
fi
echo "prep done for ${lang}"
