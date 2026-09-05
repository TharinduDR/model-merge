#!/usr/bin/env python
"""Continued pretraining (CPT) of one base model on the deduped corpus, using an
expanded (vocabulary-extended) tokenizer.

Supports three memory strategies (chosen by cpt.slurm per model size):
  full-parameter (default)   -- launch under the FSDP accelerate config.
  --peft lora                -- LoRA on all linear layers + trainable embeddings, DDP.
  --peft lora --load-in-4bit -- QLoRA (4-bit base) + LoRA + trainable embeddings, DDP.

Resume-safe: with --resume auto it continues from the latest checkpoint in --out,
and writes --done-file when it reaches max-steps, so a 24h-walled array task can be
re-submitted until it completes.
"""
from __future__ import annotations
import argparse
import glob
import os

import torch
from torch.utils.data import IterableDataset, get_worker_info


def _shard_files(files, rank, world, worker_id, num_workers):
    files = sorted(files)
    files = files[rank::world]
    files = files[worker_id::num_workers]
    return files


class PackedParquet(IterableDataset):
    def __init__(self, data_dir, tokenizer, seq_len, replay_files=None, replay_ratio=0.0):
        self.files = glob.glob(os.path.join(data_dir, "*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No .parquet in {data_dir}")
        self.replay_files = replay_files or []
        self.replay_ratio = replay_ratio
        self.tok = tokenizer
        self.seq_len = seq_len
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))

    def _iter_texts(self, files):
        import pyarrow.parquet as pq
        for path in files:
            pf = pq.ParquetFile(path)
            for i in range(pf.num_row_groups):
                for t in pf.read_row_group(i, columns=["text"]).column("text").to_pylist():
                    if t:
                        yield t

    def __iter__(self):
        import random
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        files = _shard_files(self.files, self.rank, self.world, wid, nw)
        replay = _shard_files(self.replay_files, self.rank, self.world, wid, nw) if self.replay_files else []

        eos = self.tok.eos_token_id
        buf = []
        streams = [self._iter_texts(files)]
        weights = [1.0]
        if replay and self.replay_ratio > 0:
            streams.append(self._iter_texts(replay))
            weights = [1.0 - self.replay_ratio, self.replay_ratio]

        rng = random.Random(1234 + self.rank * 97 + wid)
        exhausted = [False] * len(streams)
        while not all(exhausted):
            idx = rng.choices(range(len(streams)), weights=weights)[0]
            if exhausted[idx]:
                continue
            try:
                text = next(streams[idx])
            except StopIteration:
                exhausted[idx] = True
                continue
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            ids.append(eos)
            buf.extend(ids)
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                yield {"input_ids": chunk, "labels": list(chunk)}


def collate(batch):
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": torch.ones_like(input_ids)}


def apply_vocab_expansion(model, tokenizer, embed_init_path):
    model.resize_token_embeddings(len(tokenizer), mean_resizing=True)
    if not embed_init_path or not os.path.exists(embed_init_path):
        return
    blob = torch.load(embed_init_path, map_location="cpu")
    new_ids = blob["new_ids"]
    in_emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for row, tid in enumerate(new_ids):
            in_emb[tid] = blob["input"][row].to(in_emb.dtype)
        if blob.get("output") is not None:
            oe = model.get_output_embeddings()
            if oe is not None:
                for row, tid in enumerate(new_ids):
                    oe.weight[tid] = blob["output"][row].to(oe.weight.dtype)
    print(f"Applied vocab expansion: {len(new_ids)} new rows, vocab={len(tokenizer)}")


def maybe_wrap_peft(model, load_in_4bit, r, alpha, dropout):
    """LoRA on all linear layers; keep the (expanded) embeddings + lm_head fully
    trainable via modules_to_save so the new tokens actually learn."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, cfg)
    model.enable_input_require_grads()          # needed for grad-checkpointing + PEFT
    model.print_trainable_parameters()
    return model


def main():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from model_utils import load_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--loader", default="causal_lm")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--embed-init", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--not-base", action="store_true")
    p.add_argument("--peft", choices=["none", "lora"], default="none")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--resume", choices=["auto", "none"], default="none")
    p.add_argument("--done-file", default=None)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--replay-data", default=None)
    p.add_argument("--replay-ratio", type=float, default=0.05)
    a = p.parse_args()

    rank0 = int(os.environ.get("RANK", 0)) == 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if a.not_base and rank0:
        print("WARNING: instruct/reasoning checkpoint, not a base model -- CPT will "
              "erode instruction-following.", flush=True)

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, trust_remote_code=True)
    if tok.eos_token_id is None:
        tok.eos_token = tok.pad_token or "</s>"

    quant = None
    device_map = None
    if a.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        device_map = {"": local_rank}           # place the 4-bit shard on this rank's GPU

    model = load_model(a.model, a.loader, dtype=torch.bfloat16,
                       attn="flash_attention_2", quantization_config=quant,
                       device_map=device_map)

    if a.tokenizer:
        apply_vocab_expansion(model, tok, a.embed_init)

    model.config.use_cache = False
    if a.peft == "lora":
        model = maybe_wrap_peft(model, a.load_in_4bit, a.lora_r, a.lora_alpha, a.lora_dropout)
    else:
        model.gradient_checkpointing_enable()

    replay_files = glob.glob(os.path.join(a.replay_data, "*.parquet")) if a.replay_data else None
    train_ds = PackedParquet(a.data, tok, a.seq_len, replay_files, a.replay_ratio)

    args = TrainingArguments(
        output_dir=a.out,
        max_steps=a.max_steps,
        per_device_train_batch_size=a.micro_batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=a.warmup_ratio,
        weight_decay=0.1,
        bf16=True,
        logging_steps=20,
        save_steps=a.save_steps,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=(a.peft == "none"),
        ddp_find_unused_parameters=False,
    )

    resume = None
    if a.resume == "auto" and glob.glob(os.path.join(a.out, "checkpoint-*")):
        resume = True
        if rank0:
            print("Resuming from latest checkpoint in", a.out, flush=True)

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(a.out)
    if rank0:
        tok.save_pretrained(a.out)
        if a.done_file:
            os.makedirs(os.path.dirname(a.done_file), exist_ok=True)
            open(a.done_file, "w").write("ok\n")
            print("Wrote", a.done_file, flush=True)


if __name__ == "__main__":
    main()
