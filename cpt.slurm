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

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=
export PYTHONIOENCODING=utf-8
export HF_HUB_DOWNLOAD_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

cd "$SLURM_SUBMIT_DIR"
mkdir -p log

# ------------------------------------------------------------------------------
# Grid = langs x models. Each array task = ONE (lang, model): expand vocab, then
# continued-pretrain on this node's 3 L40S GPUs. %5 -> 5 tasks x 3 GPUs = 15 GPUs
# = your astro quota. Keep these arrays in sync with configs/models.yaml.
#
# CPT does NOT fit in 24h. The trainer checkpoints (save-steps) and resumes from
# the latest checkpoint on the next run, and writes COMPLETED when it hits
# max-steps. So: just re-run `sbatch cpt.slurm` until every task is skipped.
#
# 48GB L40S memory strategy (full Adam won't fit big models on 3x48GB):
#   full  (<=~5B)   full-parameter, FSDP full-shard across 3 GPUs, bf16
#   lora  (<=~14B)  LoRA + trainable embeddings, DDP across 3 GPUs, bf16 base
#   qlora (>=26B)   4-bit base + LoRA + trainable embeddings, DDP across 3 GPUs
# Full-parameter CPT of the >=9B models needs H200/gpu-long, not astro; override
# per model with the strategy array below if you move to that hardware.
# ------------------------------------------------------------------------------
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
# all remaining models are pretrained bases
is_base=(1 1 1 1 1  1 1 1 1 1  1 1)

n_models=${#models[@]}
i=$SLURM_ARRAY_TASK_ID
lang="${langs[$(( i / n_models ))]}"
mi=$(( i % n_models ))
model="${models[$mi]}"; loader="${loaders[$mi]}"; strat="${strategy[$mi]}"; base="${is_base[$mi]}"
name="${model##*/}"

echo "Task $i -> lang=$lang model=$model strat=$strat on $(hostname)"

DATA="data/${lang}"
SPM="tokenizers/${lang}.model"
EXP="expanded/${lang}/${name}"
OUT="runs/${lang}/${name}"
DONE="${OUT}/COMPLETED"

# Prerequisites are built by the prep step (needs internet + big scratch; not GPU).
if [ ! -d "$DATA" ] || [ ! -f "$SPM" ]; then
    echo "MISSING $DATA or $SPM -- run prep.slurm (data_pipeline + train_lang_tokenizer) first."
    exit 1
fi

# Skip entirely once the model is trained AND uploaded.
if [ -f "${OUT}/PUSHED" ]; then echo "already complete + uploaded: $lang $name -- skipping"; exit 0; fi

# 1) Train (skipped if COMPLETED already exists from an earlier finished run).
if [ ! -f "$DONE" ]; then
    # 1a) Vocabulary expansion (once per model,lang). Skipped if embed_init.pt exists.
    if [ ! -f "${EXP}/embed_init.pt" ]; then
        echo ">>> expand vocab: $model"
        python src/expand_vocab.py --model "$model" --loader "$loader" \
            --lang-spm "$SPM" --out "$EXP" --max-new "${MAX_NEW:-16000}" --sample "$DATA"
    fi

    # 1b) Continued pretraining (resume-safe).
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
fi

# 2) Once trained, export a standalone model and upload to tharindu. For lora/qlora
# this reloads the base in bf16 and merges the adapter, so the pushed model is a full,
# drop-in checkpoint (not just an adapter). Needs HF_TOKEN with WRITE access to the org.
if [ -f "$DONE" ] && [ ! -f "${OUT}/PUSHED" ]; then
    repo="tharindu/${name}-${lang}"
    echo ">>> uploading $OUT -> https://huggingface.co/$repo"
    python src/export_and_upload.py \
        --base "$model" --loader "$loader" --strategy "$strat" \
        --run-dir "$OUT" --tokenizer "$EXP" --lang "$lang" --repo-id "$repo" --public \
        && touch "${OUT}/PUSHED"
fi
