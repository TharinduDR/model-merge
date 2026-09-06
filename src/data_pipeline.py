#!/usr/bin/env python
"""Merge + deduplicate LOCAL CulturaX, HPLT 3.0, and MADLAD-400 files for one language.

Fully offline (no network). Each source's files are located in this priority order:
  1. an explicit --<source>-dir flag,
  2. <data_raw>/<source>_<code>/  (e.g. hplt_sin_Sinh, culturax_si, madlad_si),
  3. the Hugging Face cache under $HF_HOME  (for CulturaX / MADLAD downloaded with
     `huggingface-cli download` without --local-dir).
HPLT is never on the Hub, so it uses only (1) or (2).

Output: sharded Parquet under <out>/ {text, url, timestamp, source} + an <out>/_SUCCESS
sentinel written only on clean completion.

Example:
    python -u src/data_pipeline.py --language sinhala \
        --data-raw /scratch/hpc/37/ranasint/data_raw \
        --out /scratch/hpc/37/ranasint/data/sinhala --near-dedup
"""
from __future__ import annotations
import argparse
import glob
import gzip
import io
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from dedup import text_hash, NearDeduper  # noqa: E402

try:
    import orjson
    def _loads(s):
        return orjson.loads(s)
except Exception:
    def _loads(s):
        return json.loads(s)

CULTURAX_REPO = "uonlp/CulturaX"
MADLAD_REPO = "allenai/MADLAD-400"


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------- file discovery -------------------------------
def _find(dirpath, patterns):
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(dirpath, "**", pat), recursive=True)
    return sorted(set(files))


def _hf_cache_snapshot(repo):
    """Return the newest cached snapshot dir for a dataset repo, offline. None if absent."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo, repo_type="dataset", local_files_only=True)
    except Exception:
        pass
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    slug = "datasets--" + repo.replace("/", "--")
    hits = glob.glob(os.path.join(home, "**", slug, "snapshots", "*"), recursive=True)
    hits = [h for h in hits if os.path.isdir(h)]
    return sorted(hits)[-1] if hits else None


def _resolve_dir(name, code, override, data_raw, repo):
    if override:
        return override
    conv = os.path.join(data_raw, f"{name}_{code}")
    if os.path.isdir(conv):
        return conv
    if repo:
        snap = _hf_cache_snapshot(repo)
        if snap:
            return snap
    return conv


def _lang_filter(files, code):
    sep = os.sep
    kept = [f for f in files if f"{sep}{code}{sep}" in f
            or os.path.basename(f).startswith(f"{code}_")
            or f"_{code}_" in os.path.basename(f)]
    return kept or files


# ----------------------------- low-level readers ----------------------------
def _iter_jsonl_lines(path):
    if path.endswith(".zst"):
        import zstandard as zstd
        with open(path, "rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    line = line.strip()
                    if line:
                        yield line
    elif path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line
    else:
        with open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield line


# ----------------------------- source iterators -----------------------------
def iter_culturax(dirpath, code):
    import pyarrow.parquet as pq
    files = _lang_filter(_find(dirpath, ["*.parquet"]), code)
    _log(f"CulturaX: {len(files)} parquet file(s) under {dirpath}")
    want = ["text", "url", "timestamp", "source"]
    for path in files:
        pf = pq.ParquetFile(path)
        cols = [c for c in want if c in pf.schema_arrow.names]
        for i in range(pf.num_row_groups):
            b = pf.read_row_group(i, columns=cols).to_pydict()
            n = len(b.get("text", []))
            for j in range(n):
                yield {
                    "text": b["text"][j] or "",
                    "url": (b["url"][j] or "") if "url" in b else "",
                    "timestamp": str(b["timestamp"][j] or "") if "timestamp" in b else "",
                    "source": "culturax:" + (str(b["source"][j] or "") if "source" in b else ""),
                }


def iter_hplt(dirpath):
    files = _find(dirpath, ["*.jsonl.zst", "*.jsonl"])
    _log(f"HPLT: {len(files)} shard(s) under {dirpath}")
    for path in files:
        for line in _iter_jsonl_lines(path):
            try:
                r = _loads(line)
            except Exception:
                continue
            yield {
                "text": r.get("text", "") or "",
                "url": r.get("u", "") or "",
                "timestamp": str(r.get("ts", "") or ""),
                "source": "hplt3.0",
            }


def iter_madlad(dirpath, code, split="clean"):
    files = _lang_filter(_find(dirpath, ["*.jsonl.gz", "*.jsonl.zst", "*.jsonl"]), code)
    _log(f"MADLAD-400: {len(files)} file(s) under {dirpath}")
    for path in files:
        for line in _iter_jsonl_lines(path):
            try:
                r = _loads(line)
                text = r.get("text", "") if isinstance(r, dict) else str(r)
            except Exception:
                text = line
            yield {"text": text or "", "url": "", "timestamp": "", "source": f"madlad400:{split}"}


# ----------------------------- sharded writer -------------------------------
class ShardedParquetWriter:
    def __init__(self, outdir: Path, shard_docs: int = 200_000):
        import pyarrow  # noqa: F401
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
        path = self.outdir / f"part-{self._shard:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(self._buf), path, compression="zstd")
        _log(f"  wrote {path.name} ({len(self._buf):,} docs; {self.total:,} total)")
        self._shard += 1
        self._buf = []

    def close(self):
        self._flush()


# ----------------------------- driver ---------------------------------------
def build(language, cfg_path, data_raw, out, overrides, near_dedup, near_threshold,
          min_chars, shard_docs, log_every):
    cfg = yaml.safe_load(open(cfg_path))
    key = language.lower()
    if key not in cfg:
        raise SystemExit(f"'{language}' not in {cfg_path}. Have: {', '.join(cfg)}")
    lc = cfg[key]
    cx_code, hp_code, md_code = lc["culturax"], lc["hplt"], lc["madlad"]

    cx_dir = _resolve_dir("culturax", cx_code, overrides.get("culturax"), data_raw, CULTURAX_REPO)
    hp_dir = _resolve_dir("hplt", hp_code, overrides.get("hplt"), data_raw, None)
    md_dir = _resolve_dir("madlad", md_code, overrides.get("madlad"), data_raw, MADLAD_REPO)

    _log(f"language={language} out={out} near_dedup={near_dedup}")
    _log(f"culturax <- {cx_dir}")
    _log(f"hplt     <- {hp_dir}")
    _log(f"madlad   <- {md_dir}")

    planned = [
        ("culturax", cx_dir, ["*.parquet"], lambda: iter_culturax(cx_dir, cx_code)),
        ("hplt", hp_dir, ["*.jsonl.zst", "*.jsonl"], lambda: iter_hplt(hp_dir)),
        ("madlad", md_dir, ["*.jsonl.gz", "*.jsonl.zst", "*.jsonl"],
         lambda: iter_madlad(md_dir, md_code, lc.get("madlad_split", "clean"))),
    ]

    sources = []
    for name, d, pats, mk in planned:
        if d and os.path.isdir(d) and _find(d, pats):
            sources.append((name, mk))
        else:
            _log(f"WARNING: no files for {name} at {d} -- skipping it.")
    if 0 < len(sources) < 3:
        _log(f"WARNING: only {len(sources)} of 3 sources found -- corpus will be incomplete.")
    if not sources:
        raise SystemExit("No input files for any source. Check --data-raw / HF_HOME.")

    seen = set()
    near = NearDeduper(near_threshold) if near_dedup else None
    writer = ShardedParquetWriter(out, shard_docs)
    stats = {name: {"in": 0, "kept": 0} for name, _ in sources}
    t0 = time.time()

    for name, mk in sources:
        _log(f"=== source: {name} ===")
        s = stats[name]
        for rec in mk():
            s["in"] += 1
            t = rec["text"]
            if t and len(t.strip()) >= min_chars:
                h = text_hash(t)
                if h not in seen:
                    if near is not None and near.is_duplicate(t):
                        seen.add(h)
                    else:
                        seen.add(h)
                        writer.write(rec)
                        s["kept"] += 1
            if s["in"] % log_every == 0:
                el = time.time() - t0
                _log(f"  [{name}] read={s['in']:,} kept={s['kept']:,} "
                     f"({s['in']/max(el,1e-9):,.0f} docs/s, {el:,.0f}s)")
        _log(f"[{name}] DONE read={s['in']:,} kept={s['kept']:,}")

    writer.close()
    open(os.path.join(out, "_SUCCESS"), "w").write("ok\n")
    _log(f"Wrote {writer.total:,} unique docs -> {out}  (total {time.time()-t0:,.0f}s)")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", required=True)
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "configs" / "languages.yaml"))
    p.add_argument("--data-raw", default="/scratch/hpc/37/ranasint/data_raw")
    p.add_argument("--out", required=True)
    p.add_argument("--culturax-dir", default=None)
    p.add_argument("--hplt-dir", default=None)
    p.add_argument("--madlad-dir", default=None)
    p.add_argument("--near-dedup", action="store_true")
    p.add_argument("--near-threshold", type=float, default=0.8)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--shard-docs", type=int, default=200_000)
    p.add_argument("--log-every", type=int, default=10_000)
    a = p.parse_args()

    overrides = {"culturax": a.culturax_dir, "hplt": a.hplt_dir, "madlad": a.madlad_dir}
    build(a.language, a.config, a.data_raw, a.out, overrides, a.near_dedup,
          a.near_threshold, a.min_chars, a.shard_docs, a.log_every)


if __name__ == "__main__":
    main()