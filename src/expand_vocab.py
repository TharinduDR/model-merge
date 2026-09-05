#!/usr/bin/env python
"""Expand ONE base model's tokenizer with language-specific tokens, then resize and
smart-initialize the new embedding rows. Runs per (model, language).

Pipeline:
  1. Read candidate pieces from the language SentencePiece vocab.
  2. Keep only pieces the base tokenizer *doesn't already* encode as a single token
     (no point adding what's already covered -- Gemma's 256k vocab may cover a lot).
  3. add_tokens() them (works across BPE/SentencePiece/tiktoken alike).
  4. For each new token, initialise its embedding as the MEAN of the base-tokenizer
     sub-pieces of its surface string (far better than random / global-mean init).
  5. Save: extended tokenizer + embed_init.pt (new rows only) + meta.json with the
     fertility (tokens/char) before vs after, so you can see the win.

We save only the *new* embedding rows, not a full model copy, to avoid duplicating
tens of GB per (model, language) pair. continued_pretrain.py re-applies them.

Example:
    python src/expand_vocab.py --model Qwen/Qwen3.5-4B-Base --loader causal_lm \
        --lang-spm tokenizers/sinhala.model --out expanded/sinhala/Qwen3.5-4B-Base \
        --max-new 16000 --sample data/sinhala
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path


# ----------------------- pure, unit-testable helpers ------------------------
def read_spm_pieces(spm_vocab_path):
    """Return pieces from a SentencePiece .vocab file, in-order (most frequent first),
    skipping control/byte-fallback pieces."""
    pieces = []
    with open(spm_vocab_path, encoding="utf-8") as f:
        for line in f:
            piece = line.rstrip("\n").split("\t")[0]
            if not piece or piece in ("<unk>", "<s>", "</s>", "<pad>"):
                continue
            if piece.startswith("<") and piece.endswith(">"):   # <0x41> byte pieces etc.
                continue
            pieces.append(piece)
    return pieces


def piece_to_surface(piece: str) -> str:
    """SentencePiece marks a leading space with U+2581. Convert to a real string:
    '▁word' -> ' word', 'word' -> 'word'."""
    return piece.replace("\u2581", " ")


def select_new_tokens(pieces, base_tok, max_new: int, min_len: int = 2):
    """Surfaces that the base tokenizer splits into >=2 tokens (i.e. not already covered),
    de-duplicated, capped at max_new. Returns list of surface strings."""
    new, seen = [], set()
    for p in pieces:
        s = piece_to_surface(p)
        stripped = s.strip()
        if len(stripped) < min_len or s in seen:
            continue
        # already a single token in the base vocab? skip.
        if len(base_tok.encode(s, add_special_tokens=False)) <= 1:
            continue
        seen.add(s)
        new.append(s)
        if len(new) >= max_new:
            break
    return new


def mean_init_rows(surfaces, old_tok, in_emb, out_emb=None):
    """For each surface string, mean the base-tokenizer sub-piece rows of `in_emb`
    (and `out_emb` if untied). Returns (in_new [N,d], out_new [N,d] or None)."""
    import torch

    d = in_emb.shape[1]
    in_new = torch.empty(len(surfaces), d, dtype=in_emb.dtype)
    out_new = None if out_emb is None else torch.empty(len(surfaces), d, dtype=out_emb.dtype)
    in_mean = in_emb.mean(0)
    out_mean = None if out_emb is None else out_emb.mean(0)

    for i, s in enumerate(surfaces):
        ids = old_tok.encode(s, add_special_tokens=False)
        ids = [j for j in ids if j < in_emb.shape[0]]
        if ids:
            in_new[i] = in_emb[ids].mean(0)
            if out_new is not None:
                out_new[i] = out_emb[ids].mean(0)
        else:                                   # fallback: global mean
            in_new[i] = in_mean
            if out_new is not None:
                out_new[i] = out_mean
    return in_new, out_new


def fertility(tok, texts):
    """Mean tokens-per-character over `texts` (lower is better)."""
    ntok = nchar = 0
    for t in texts:
        if not t:
            continue
        ntok += len(tok.encode(t, add_special_tokens=False))
        nchar += len(t)
    return (ntok / nchar) if nchar else float("nan")


def _sample_texts(data_dir, n=2000):
    import pyarrow.parquet as pq

    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
        pf = pq.ParquetFile(path)
        for t in pf.read_row_group(0, columns=["text"]).column("text").to_pylist():
            if t:
                out.append(t)
            if len(out) >= n:
                return out
    return out


# --------------------------------- driver -----------------------------------
def main():
    import torch
    from transformers import AutoTokenizer
    from tokenizers import AddedToken

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from src.model_utils import load_model, output_embeddings_or_none

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--loader", default="causal_lm")
    p.add_argument("--lang-spm", required=True, help="Path to <lang>.model from the tokenizer step")
    p.add_argument("--out", required=True)
    p.add_argument("--max-new", type=int, default=16000)
    p.add_argument("--sample", default=None, help="Parquet dir to measure fertility before/after")
    a = p.parse_args()

    spm_vocab = a.lang_spm.replace(".model", ".vocab")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    old_tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    ext_tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    pieces = read_spm_pieces(spm_vocab)
    new_surfaces = select_new_tokens(pieces, old_tok, a.max_new)
    if not new_surfaces:
        print("Base tokenizer already covers this language well; nothing to add.")

    sample = _sample_texts(a.sample) if a.sample else []
    fert_before = fertility(old_tok, sample) if sample else None

    ext_tok.add_tokens([AddedToken(s, normalized=False, special=False) for s in new_surfaces])
    new_ids = [ext_tok.convert_tokens_to_ids(s) for s in new_surfaces]

    # Load model, grab original embeddings, resize, then overwrite the new rows.
    model = load_model(a.model, a.loader, dtype=torch.float32, low_cpu_mem_usage=True)
    in_emb = model.get_input_embeddings().weight.data.clone()
    out_emb_t = output_embeddings_or_none(model)
    out_emb = out_emb_t.data.clone() if out_emb_t is not None else None

    in_new, out_new = mean_init_rows(new_surfaces, old_tok, in_emb, out_emb)

    fert_after = fertility(ext_tok, sample) if sample else None

    torch.save({"new_ids": new_ids, "input": in_new, "output": out_new},
               out / "embed_init.pt")
    ext_tok.save_pretrained(out)
    meta = {
        "model": a.model,
        "base_vocab": len(old_tok),
        "added": len(new_surfaces),
        "new_vocab": len(ext_tok),
        "tied_embeddings": out_new is None,
        "fertility_tokens_per_char_before": fert_before,
        "fertility_tokens_per_char_after": fert_after,
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
