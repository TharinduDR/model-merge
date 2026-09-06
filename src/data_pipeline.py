#!/usr/bin/env python
"""Merge + deduplicate LOCAL CulturaX, HPLT 3.0, and MADLAD-400 files for one language.

Fully offline. Files are found by scanning for the ACTUAL files, in priority order:
  1. an explicit --<source>-dir flag,
  2. <data_raw>/<source>_<code>/ if it contains matching files,
  3. the Hugging Face cache: ALL snapshots of the dataset under $HF_HOME (and ~/.cache),
     matched case-insensitively -- a stale metadata-only snapshot is skipped and the
     snapshot that actually holds the data is used.
HPLT is never on the Hub, so it uses only (1) or (2).
MADLAD is filtered to the split in languages.yaml (madlad_split, default "clean"),
so a noisy shard sitting next to the clean one is ignored.

Output: sharded Parquet under <out>/ {text, url, timestamp, source} + <out>/_SUCCESS.

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
CX_PATS = ["*.parquet"]
HP_PATS = ["*.jsonl.zst", "*.jsonl"]
MD_PATS = ["*.jsonl.gz", "*.jsonl.zst", "*.jsonl"]


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _find(dirpath, patterns):
    out = []
    for pat in patterns:
        out += glob.glob(os.path.join(dirpath, "**", pat), recursive=True)
    return sorted(set(out))


def _cache_files(repo, patterns):
    """Every matching file across ALL cached snapshots of `repo`, case-insensitive."""
    name = repo.split("/")[-1].lower()
    homes = {os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface")}
    files = []
    for home in filter(None, homes):
        for d in glob.glob(os.path.join(home, "**", "datasets--*"), recursive=True):
            if os.path.isdir(d) and name in os.path.basename(d).lower():
                files += _find(d, patterns)
    return sorted(set(files))


def _lang_filter(files, code):
    sep = os.sep
    kept = [f for f in files if f"{sep}{code}{sep}" in f
            or os.path.basename(f).startswith(f"{code}_")
            or f"_{code}_" in os.path.basename(f)]
    return kept or files


def _resolve_files(name, code, override, data_raw, repo, patterns, lang_scope):
    if override:
        files = _find(override, patterns)
        return _lang_filter(files, code) if lang_scope else files
    conv = os.path.join(data_raw, f"{name}_{code}")
    files = _find(conv, patterns) if os.path.isdir(conv) else []
    if files:
        return _lang_filter(files, code) if lang_scope else files
    if repo:
        return _lang_filter(_cache_files(repo, patterns), code)
    return []


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
def iter_culturax(files):
    import pyarrow.parquet as pq
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


def iter_hplt(files):
    for path in files:
        for line in _iter_jsonl_lines(path):
            try:
                r = _loads(line)
            except Exception:
                continue
            yield {"text": r.get("text", "") or "", "url": r.get("u", "") or "",
                   "timestamp": str(r.get("ts", "") or ""), "source": "hplt3.0"}


def iter_madlad(files, split="clean"):
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
    md_split = lc.get("madlad_split", "clean")

    cx = _resolve_files("culturax", cx_code, overrides.get("culturax"), data_raw, CULTURAX_REPO, CX_PATS, True)
    hp = _resolve_files("hplt", hp_code, overrides.get("hplt"), data_raw, None, HP_PATS, False)
    md = _resolve_files("madlad", md_code, overrides.get("madlad"), data_raw, MADLAD_REPO, MD_PATS, True)

    # Keep only the requested MADLAD split (e.g. clean), dropping noisy shards.
    md_kept = [f for f in md if md_split in os.path.basename(f)]
    if md and not md_kept:
        _log(f"WARNING: MADLAD split '{md_split}' not in any filename; using all {len(md)} file(s).")
        md_kept = md
    md = md_kept

    _log(f"language={language} out={out} near_dedup={near_dedup}")
    _log(f"culturax: {len(cx)} file(s)")
    _log(f"hplt:     {len(hp)} file(s)")
    _log(f"madlad:   {len(md)} file(s) (split={md_split})")
    for tag, fl in (("culturax", cx), ("hplt", hp), ("madlad", md)):
        if fl:
            _log(f"  {tag} e.g. {fl[0]}")

    sources = [(n, mk) for n, fl, mk in
               (("culturax", cx, lambda: iter_culturax(cx)),
                ("hplt", hp, lambda: iter_hplt(hp)),
                ("madlad", md, lambda: iter_madlad(md, md_split)))
               if fl]
    for m in ({"culturax", "hplt", "madlad"} - {n for n, _ in sources}):
        _log(f"WARNING: no files found for {m}.")
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