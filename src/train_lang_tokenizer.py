#!/usr/bin/env python
"""Train a language-specific SentencePiece BPE tokenizer on the deduped corpus.

FIX: feed SentencePiece one LINE per training unit, not a whole document. Whole
documents blew past max_sentence_length (default 4192) so most were skipped as
"too long", and their internal newlines produced pieces like ".\\n" / "2019\\n"
that break the .vocab format and corrupt downstream .vocab parsing. Splitting on
newlines fixes both. max_sentence_length is also raised for any genuinely long line.

Run once per language.

Example:
    python -u src/train_lang_tokenizer.py --data /scratch/hpc/37/ranasint/data/sinhala \
        --out /scratch/hpc/37/ranasint/tokenizers/sinhala
"""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path


def line_iter(files, max_bytes, min_line_chars=8):
    """Yield individual lines from Parquet 'text' fields until max_bytes is read.
    One line per training unit keeps each unit short and newline-free."""
    import pyarrow.parquet as pq

    seen = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for i in range(pf.num_row_groups):
            for doc in pf.read_row_group(i, columns=["text"]).column("text").to_pylist():
                if not doc:
                    continue
                for line in doc.split("\n"):
                    line = line.strip()
                    if len(line) < min_line_chars:
                        continue
                    yield line
                    seen += len(line.encode("utf-8"))
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
    p.add_argument("--max-sentence-length", type=int, default=16384,
                   help="Per-line byte cap; lines longer than this are skipped by SP.")
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet in {a.data} -- run data_pipeline.py first.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        sentence_iterator=line_iter(files, int(a.max_gb * 1e9)),
        model_prefix=a.out,
        vocab_size=a.vocab_size,
        model_type="bpe",
        character_coverage=a.character_coverage,
        normalization_rule_name="nfkc",
        max_sentence_length=a.max_sentence_length,
        train_extremely_large_corpus=True,
        num_threads=os.cpu_count() or 4,
        input_sentence_size=10_000_000,
        shuffle_input_sentence=True,
        byte_fallback=True,
        remove_extra_whitespaces=True,
    )
    print(f"Wrote {a.out}.model / {a.out}.vocab")


if __name__ == "__main__":
    main()