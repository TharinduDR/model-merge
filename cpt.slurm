#!/bin/bash
#SBATCH -p astro
#SBATCH --array=0-11%5
#SBATCH --gres=gpu:nvidia_l40s:3
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=log/cpt_%A_%a.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

# --- environment first (while nounset is OFF, so /etc/profile.d doesn't trip it) ---
source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0
source activate /storage/hpc/37/ranasint/conda_envs/llm_exp

set -euo pipefail

SCRATCH=/scratch/hpc/37/ranasint
export HF_HOME="${SCRATCH}/hf_cache"          # base models are read from this cache
export HF_TOKEN=
export HF_HUB_OFFLINE=1                        # compute nodes are air-gapped: read cache only
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

cd "$SLURM_SUBMIT_DIR"; mkdir -p log

# All training artefacts live on scratch (submit dir is space-limited).
DATA_ROOT="${SCRATCH}/data"
TOK_ROOT="${SCRATCH}/tokenizers"
EXP_ROOT="${SCRATCH}/expanded"
RUN_ROOT="${SCRATCH}/runs"

# --- model grid (keep in sync with configs/models.yaml) ---
langs=(sinhala)

models=(Qwen/Qwen3.5-27B Qwen/Qwen3.5-9B-Base Qwen/Qwen3.5-4B-Base Qwen/Qwen3.5-2B-Base Qwen/Qwen3.5-0.8B-Base \
        google/gemma-4-31B google/gemma-4-26B-A4B google/gemma-4-12B google/gemma-4-E4B google/gemma-4-E2B \
        meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-1B)
loaders=(causal_lm causal_lm causal_lm causal_lm causal_lm \
         image_text_to_text image_text_to_text image_text_to_text image_text_to_text image_text_to_text \
         causal_lm causal_lm)
strategy=(qlora lora full full full \
          qlora qlora lora lora full \
          full full)
is_base=(1 1 1 1 1  1 1 1 1 1  1 1)

n_models=${#models[@]}
i=$SLURM_ARRAY_TASK_ID
lang="${langs[$(( i / n_models ))]}"
mi=$(( i % n_models ))
model="${models[$mi]}"; loader="${loaders[$mi]}"; strat="${strategy[$mi]}"; base="${is_base[$mi]}"
name="${model##*/}"

echo "Task $i -> lang=$lang model=$model strat=$strat on $(hostname)"

DATA="${DATA_ROOT}/${lang}"
SPM="${TOK_ROOT}/${lang}.model"
EXP="${EXP_ROOT}/${lang}/${name}"
OUT="${RUN_ROOT}/${lang}/${name}"
DONE="${OUT}/COMPLETED"

# Prerequisites produced by prep.slurm.
[ -f "${DATA}/_SUCCESS" ] || { echo "MISSING ${DATA}/_SUCCESS -- run prep first."; exit 1; }
[ -f "${SPM}" ]           || { echo "MISSING ${SPM} -- run prep first."; exit 1; }

# Skip once trained.
if [ -f "$DONE" ]; then echo "already complete: $lang $name -- skipping"; exit 0; fi

# 1) Vocabulary expansion (once per model,lang). Loads the base model from the HF cache.
if [ ! -f "${EXP}/embed_init.pt" ]; then
    echo ">>> expand vocab: $model"
    python -u src/expand_vocab.py --model "$model" --loader "$loader" \
        --lang-spm "$SPM" --out "$EXP" --max-new "${MAX_NEW:-16000}" --sample "$DATA"
fi

# 2) Continued pretraining (resume-safe).
notbase=""; [ "$base" = "0" ] && notbase="--not-base"
port=$(( 29500 + SLURM_ARRAY_TASK_ID ))

common=( --model "$model" --loader "$loader" $notbase \
         --tokenizer "$EXP" --embed-init "${EXP}/embed_init.pt" \
         --data "$DATA" --out "$OUT" --resume auto --done-file "$DONE" \
         --seq-len 4096 --save-steps 500 --max-steps "${MAX_STEPS:-20000}" )

case "$strat" in
  full)
    accelerate launch --config_file configs/accelerate_fsdp.yaml \
        --num_processes 3 --main_process_port $port \
        src/continued_pretrain.py "${common[@]}" --micro-batch 2 --grad-accum 16
    ;;
  lora)
    accelerate launch --config_file configs/accelerate_ddp.yaml \
        --num_processes 3 --main_process_port $port \
        src/continued_pretrain.py "${common[@]}" --peft lora --micro-batch 2 --grad-accum 16
    ;;
  qlora)
    accelerate launch --config_file configs/accelerate_ddp.yaml \
        --num_processes 3 --main_process_port $port \
        src/continued_pretrain.py "${common[@]}" --peft lora --load-in-4bit --micro-batch 1 --grad-accum 32
    ;;
esac

echo "training finished for $lang $name; upload with upload_all.sh on a login node."