#!/usr/bin/env python
"""Turn a finished CPT run into a standalone model repo and push it to the Hub.

- full   : the run dir is already a full model -> load + push.
- lora    : reload base in bf16, resize to the extended tokenizer, attach the adapter
            (which also carries the trained embeddings via modules_to_save), merge, push.
- qlora   : same as lora -- we reload the base in bf16 (NOT 4-bit) so it merges cleanly.

Runs as a single process (no accelerate) after training completes. Idempotency is
handled by the caller (a PUSHED sentinel).

Example:
    python src/export_and_upload.py --base Qwen/Qwen3.5-4B-Base --loader causal_lm \
        --strategy lora --run-dir runs/sinhala/Qwen3.5-4B-Base \
        --tokenizer expanded/sinhala/Qwen3.5-4B-Base --lang sinhala \
        --repo-id tharindu/Qwen3.5-4B-Base-sinhala
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model  # noqa: E402

# Base-family -> license id + any attribution the license requires in derivatives.
LICENSE = {
    "qwen": ("apache-2.0", ""),
    "gemma": ("gemma", "This model is a derivative of Google Gemma and is subject to the "
                        "Gemma Terms of Use; the name retains the \"gemma\" prefix as required."),
    "llama": ("llama3.2", "Built with Llama. This model is a derivative of Meta Llama 3.2 and is "
                          "subject to the Llama 3.2 Community License; the name retains \"Llama\" "
                          "as required."),
}

# Language name (as used on the CLI / in configs) -> ISO 639-1/3 code for the model-card
# front-matter. Add rows as you add languages; unknown names fall back to the name itself.
LANG_CODE = {
    "sinhala": "si",
    "marathi": "mr",
    "tamil": "ta",
    "afrikaans": "af",
    "nepali": "ne",
}


def family(base_repo: str) -> str:
    b = base_repo.lower()
    if "qwen" in b:
        return "qwen"
    if "gemma" in b:
        return "gemma"
    if "llama" in b:
        return "llama"
    return "qwen"


def model_card(repo_id, base, lang, exp_dir):
    lic, notice = LICENSE[family(base)]
    code = LANG_CODE.get(lang.lower(), lang.lower())
    meta = {}
    mpath = os.path.join(os.path.dirname(run_dir.rstrip("/")), "")  # noqa
    exp_meta = os.path.join(exp_dir, "meta.json")
    if os.path.exists(exp_meta):
        meta = json.load(open(exp_meta))
    added = meta.get("added", "several thousand")
    fb = meta.get("fertility_tokens_per_char_before")
    fa = meta.get("fertility_tokens_per_char_after")
    fert = ""
    if fb and fa:
        fert = (f"Tokenizer fertility on {lang.capitalize()} dropped from "
                f"{fb:.3f} to {fa:.3f} tokens/char after vocabulary expansion.\n\n")
    front = (
        "---\n"
        f"language:\n- {code}\n"
        f"license: {lic}\n"
        f"base_model: {base}\n"
        "library_name: transformers\n"
        f"tags:\n- continued-pretraining\n- vocabulary-expansion\n- {lang.lower()}\n- multilingual\n"
        "datasets:\n- uonlp/CulturaX\n- HPLT/HPLT3.0\n- allenai/MADLAD-400\n"
        "pipeline_tag: text-generation\n"
        "---\n\n"
    )
    body = (
        f"# {repo_id.split('/')[-1]}\n\n"
        f"`{base}` continued-pretrained on {lang.capitalize()} text, with a vocabulary "
        f"expanded by ~{added} {lang.capitalize()} tokens for tokenization efficiency.\n\n"
        f"{fert}"
        "## Data\n\n"
        "Deduplicated union of the Sinhala portions of CulturaX, HPLT 3.0, and MADLAD-400 "
        "(exact + MinHash near-dedup across the three sources).\n\n"
        "## Method\n\n"
        "A SentencePiece tokenizer was trained on the corpus; pieces not already covered by "
        "the base tokenizer were added, and their embeddings initialised as the mean of the "
        "base tokenizer's sub-pieces before continued pretraining.\n\n"
        "## Intended use and limitations\n\n"
        "A base (not instruction-tuned) model for Sinhala text generation and as a starting "
        "point for downstream fine-tuning. It may reflect biases and noise present in web data.\n\n"
        "## License\n\n"
        f"Released under `{lic}`, inherited from the base model. {notice}\n"
    )
    return front + body


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--loader", default="causal_lm")
    p.add_argument("--strategy", choices=["full", "lora", "qlora"], required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tokenizer", required=True, help="Extended tokenizer dir.")
    p.add_argument("--lang", required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--public", action="store_true", help="Default is a private repo.")
    a = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set (needs WRITE access to the tharindu account).")

    from transformers import AutoTokenizer
    from huggingface_hub import create_repo, upload_file

    tok_dir = a.tokenizer if os.path.exists(os.path.join(a.tokenizer, "tokenizer_config.json")) else a.run_dir
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    if a.strategy == "full":
        model = load_model(a.run_dir, a.loader, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    else:
        from peft import PeftModel
        model = load_model(a.base, a.loader, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        model.resize_token_embeddings(len(tok))     # shapes must match the trained adapter
        model = PeftModel.from_pretrained(model, a.run_dir)
        model = model.merge_and_unload()
        print("Merged adapter into base.")

    create_repo(a.repo_id, token=token, private=not a.public, exist_ok=True, repo_type="model")
    model.push_to_hub(a.repo_id, token=token, safe_serialization=True, max_shard_size="5GB")
    tok.push_to_hub(a.repo_id, token=token)

    card = model_card(a.repo_id, a.base, a.lang, a.tokenizer)
    upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                repo_id=a.repo_id, token=token, repo_type="model")
    print(f"Uploaded {a.repo_id}")


if __name__ == "__main__":
    main()
