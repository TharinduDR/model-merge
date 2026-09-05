#!/usr/bin/env python
"""Train a language-specific SentencePiece BPE tokenizer on the deduped corpus.

The pieces it learns become the *candidate* new tokens for vocabulary expansion.
Run once per language; the result is shared by every model's expansion step.

Example:
    python src/train_lang_tokenizer.py --data data/sinhala \
        --out tokenizers/sinhala --vocab-size 32000 --max-gb 4
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path


def text_iter(files, max_bytes):
    """Stream text from Parquet row groups until max_bytes is reached."""
    import pyarrow.parquet as pq

    seen = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            for t in pf.read_row_group(i, columns=["text"]).column("text").to_pylist():
                if not t:
                    continue
                yield t
                seen += len(t.encode("utf-8"))
                if seen >= max_bytes:
                    return


def main():
    import sentencepiece as spm

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Parquet dir from data_pipeline.py")
    p.add_argument("--out", required=True, help="Output prefix, e.g. tokenizers/sinhala")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-gb", type=float, default=4.0,
                   help="Cap on training bytes (SP trains fine on a few GB sample).")
    p.add_argument("--character-coverage", type=float, default=0.9995)
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet in {a.data} -- run data_pipeline.py first.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        sentence_iterator=text_iter(files, int(a.max_gb * 1e9)),
        model_prefix=a.out,
        vocab_size=a.vocab_size,
        model_type="bpe",
        character_coverage=a.character_coverage,
        normalization_rule_name="nfkc",
        train_extremely_large_corpus=True,
        num_threads=os.cpu_count() or 4,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        byte_fallback=True,          # so any rare glyph still round-trips
    )
    print(f"Wrote {a.out}.model / {a.out}.vocab")


if __name__ == "__main__":
    main()
