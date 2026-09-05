#!/usr/bin/env python
"""Download + merge + deduplicate CulturaX, HPLT 3.0, and MADLAD-400 for one language.

Output: sharded Parquet under <out>/ with schema {text, url, timestamp, source}.

Auth: CulturaX and MADLAD-400 are gated. Accept their terms on the Hub, then either
`huggingface-cli login` or pass --hf-token / set HF_TOKEN. HPLT 3.0 needs no token
(it is fetched directly from data.hplt-project.org, not the Hub).

Example:
    python src/data_pipeline.py --language sinhala --out data/sinhala --near-dedup
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from dedup import text_hash, NearDeduper  # noqa: E402


# ----------------------------- source iterators -----------------------------
def iter_culturax(code, token, streaming=True):
    from datasets import load_dataset

    ds = load_dataset("uonlp/CulturaX", code, split="train",
                      streaming=streaming, token=token)
    for r in ds:
        yield {
            "text": r.get("text", ""),
            "url": r.get("url", "") or "",
            "timestamp": str(r.get("timestamp", "") or ""),
            "source": f"culturax:{r.get('source', '')}",
        }


def iter_madlad(code, token, split="clean", streaming=True):
    from datasets import load_dataset

    ds = load_dataset("allenai/madlad-400", code, split=split,
                      streaming=streaming)
    for r in ds:
        yield {
            "text": r.get("text", ""),
            "url": "",
            "timestamp": "",
            "source": f"madlad400:{split}",
        }


def iter_hplt(code, streaming=True):
    """HPLT 3.0 is distributed as .jsonl.zst shards listed in a per-language .map file.
    We fetch the map, then let `datasets` stream the shard URLs directly."""
    import requests
    from datasets import load_dataset

    url = f"https://data.hplt-project.org/three/sorted/{code}.map"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    shard_urls = [u for u in resp.text.strip().split("\n") if u.strip()]
    if not shard_urls:
        raise RuntimeError(f"HPLT map for {code} was empty: {url}")

    ds = load_dataset("json", data_files=shard_urls, split="train", streaming=streaming)
    for rec in ds:
        yield {
            "text": rec.get("text", "") or "",
            "url": rec.get("u", "") or "",
            "timestamp": str(rec.get("ts", "") or ""),
            "source": "hplt3.0",
        }


# ----------------------------- sharded writer -------------------------------
class ShardedParquetWriter:
    def __init__(self, outdir: Path, shard_docs: int = 200_000):
        import pyarrow as pa  # noqa: F401

        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.shard_docs = shard_docs
        self._buf = []
        self._shard = 0
        self.total = 0

    def write(self, rec):
        self._buf.append(rec)
        self.total += 1
        if len(self._buf) >= self.shard_docs:
            self._flush()

    def _flush(self):
        if not self._buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self._buf)
        path = self.outdir / f"part-{self._shard:05d}.parquet"
        pq.write_table(table, path, compression="zstd")
        self._shard += 1
        self._buf = []

    def close(self):
        self._flush()


# ----------------------------- driver ---------------------------------------
def build(language, cfg_path, out, token, near_dedup, near_threshold,
          min_chars, shard_docs, streaming):
    cfg = yaml.safe_load(open(cfg_path))
    key = language.lower()
    if key not in cfg:
        raise SystemExit(f"'{language}' not in {cfg_path}. Have: {', '.join(cfg)}")
    lc = cfg[key]

    sources = [
        ("culturax", iter_culturax(lc["culturax"], token, streaming)),
        ("hplt", iter_hplt(lc["hplt"], streaming)),
        ("madlad", iter_madlad(lc["madlad"], token,
                               lc.get("madlad_split", "clean"), streaming)),
    ]

    seen = set()                       # exact-dup SHA1s
    near = NearDeduper(near_threshold) if near_dedup else None
    writer = ShardedParquetWriter(out, shard_docs)
    stats = {name: {"in": 0, "kept": 0} for name, _ in sources}

    for name, it in sources:
        for rec in it:
            stats[name]["in"] += 1
            t = rec["text"]
            if not t or len(t.strip()) < min_chars:
                continue
            h = text_hash(t)
            if h in seen:
                continue
            if near is not None and near.is_duplicate(t):
                seen.add(h)
                continue
            seen.add(h)
            writer.write(rec)
            stats[name]["kept"] += 1
        print(f"[{name}] read={stats[name]['in']:,} kept={stats[name]['kept']:,}",
              flush=True)

    writer.close()
    print(f"\nWrote {writer.total:,} unique docs -> {out}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", required=True)
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "configs" / "languages.yaml"))
    p.add_argument("--out", required=True)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--near-dedup", action="store_true",
                   help="Enable MinHash-LSH near-dedup (in-memory; single-box scale).")
    p.add_argument("--near-threshold", type=float, default=0.8)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--shard-docs", type=int, default=200_000)
    p.add_argument("--no-streaming", action="store_true",
                   help="Materialize each dataset instead of streaming (needs disk).")
    a = p.parse_args()

    build(a.language, a.config, a.out, a.hf_token, a.near_dedup, a.near_threshold,
          a.min_chars, a.shard_docs, streaming=not a.no_streaming)


if __name__ == "__main__":
    main()
