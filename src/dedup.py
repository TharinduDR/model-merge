"""Text normalization + exact and near-duplicate detection.

Exact dedup streams (one SHA1 per doc, kept in a set).
Near-dedup uses MinHash + LSH over word shingles.

For corpora in the tens-of-millions of docs and up, an in-memory `set`/LSH
will not fit. In that regime use a purpose-built tool instead:
  - datatrove  (HuggingFace) MinHash pipeline, or
  - text-dedup (Spark MinHashLSH).
The functions here are the same algorithm at a scale you can run on one box.
"""
from __future__ import annotations
import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC + whitespace collapse. Deliberately case-preserving:
    most target languages here are non-Latin scripts where casing is irrelevant,
    and lowercasing Latin content would merge genuinely distinct docs."""
    text = unicodedata.normalize("NFKC", text)
    return _WS.sub(" ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 5):
    tokens = normalize_text(text).split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def make_minhash(text: str, num_perm: int = 128, k: int = 5):
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text, k):
        m.update(sh.encode("utf-8"))
    return m


class NearDeduper:
    """MinHash-LSH near-duplicate filter. `is_duplicate` returns True if a
    document within `threshold` Jaccard similarity has already been seen."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 128, k: int = 5):
        from datasketch import MinHashLSH

        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm
        self.k = k
        self._i = 0

    def is_duplicate(self, text: str) -> bool:
        m = make_minhash(text, self.num_perm, self.k)
        if self.lsh.query(m):
            return True
        self.lsh.insert(f"d{self._i}", m)
        self._i += 1
        return False
