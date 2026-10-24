"""Knowledge-base loading, chunking and hybrid (dense + BM25) retrieval.

Design notes
------------
* Chunks follow the documents' own [SECTION-ID] markers so retrieval can be checked
  against `gold_evidence`; status / effective_from are kept on every chunk.
* Source precedence (brief): current policy -> agent guide -> help -> FAQ -> marketing.
  Superseded documents (e.g. pol_returns_v3_superseded.md) are indexed but excluded
  from retrieval unless explicitly requested.
* Dense embeddings are optional: if sentence-transformers is missing (or EMBEDDING_MODEL=none)
  the retriever silently falls back to BM25, which is still strong on small policy docs.
"""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from . import config

TIER = {"pol_": 1, "agent_": 2, "help_": 3, "faq_": 4, "mkt_": 5}
_SECTION_RE = re.compile(r"^\s*(?:#{1,6}\s*)?\[([A-Za-z0-9][A-Za-z0-9_.\-]*)\]\s*(.*)$")
_HEADING_RE = re.compile(r"^\s*#{1,3}\s+(.*)$")
_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+")
MAX_CHARS = 1400


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _front_matter(text):
    meta, body = {}, text
    if text.lstrip().startswith("---"):
        parts = text.lstrip().split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip().lower()] = v.strip().strip("'\"")
            body = parts[2]
    # tolerate "status: current" style lines in the first few body lines
    for line in body.splitlines()[:25]:
        m = re.match(r"^\W*(status|effective_from|version)\W*[:=]\s*(.+)$", line.strip(), re.I)
        if m:
            meta.setdefault(m.group(1).lower(), m.group(2).strip().strip("*_ `'\""))
    return meta, body


def _tier(filename):
    for prefix, t in TIER.items():
        if filename.startswith(prefix):
            return t
    return 9


def _split_long(text):
    if len(text) <= MAX_CHARS:
        return [text]
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > MAX_CHARS:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return out


def load_chunks(kb_dir=None):
    kb_dir = Path(kb_dir or config.KB_DIR)
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        meta, body = _front_matter(raw)
        status = meta.get("status", "").lower()
        superseded = "supersed" in status or "supersed" in path.stem.lower()
        sections, cur_id, cur_title, cur_lines = [], None, "", []

        def flush():
            text = "\n".join(cur_lines).strip()
            body_only = re.sub(r"^\s*#+.*$|^\s*\[[^\]]+\].*$", "", text, flags=re.M).strip() if not cur_id else text
            if text and (cur_id or len(body_only) > 20):  # drop heading-only fragments
                sections.append((cur_id, cur_title, text))

        for line in body.splitlines():
            m = _SECTION_RE.match(line)
            h = _HEADING_RE.match(line)
            if m:
                flush()
                cur_id, cur_title, cur_lines = m.group(1), m.group(2).strip(), [line]
            elif h and not cur_id:
                flush()
                cur_id, cur_title, cur_lines = None, h.group(1).strip(), [line]
            else:
                cur_lines.append(line)
        flush()
        for s_idx, (sec_id, title, text) in enumerate(sections):
            for p_idx, piece in enumerate(_split_long(text)):
                cid = f"{path.stem}#{sec_id or s_idx}" + (f".{p_idx}" if p_idx else "")
                chunks.append({
                    "id": cid,
                    "section_id": sec_id or "",
                    "source": path.name,
                    "title": title,
                    "status": status or "current",
                    "effective_from": meta.get("effective_from", ""),
                    "tier": _tier(path.name),
                    "superseded": superseded,
                    # keep file/title in the indexed text so "returns" queries find pol_returns
                    "text": piece,
                    "index_text": f"{path.stem.replace('_', ' ')} {title}\n{piece}",
                })
    return chunks


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(c.values()) or 1 for c in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        q = tokenize(query)
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


def _minmax(xs):
    lo, hi = min(xs), max(xs)
    return [0.0 if hi == lo else (x - lo) / (hi - lo) for x in xs]


class KnowledgeBase:
    def __init__(self, kb_dir=None, embedding_model=None):
        self.chunks = load_chunks(kb_dir)
        if not self.chunks:
            raise RuntimeError(f"No knowledge-base chunks found in {kb_dir or config.KB_DIR}. "
                               "Set DATASET_DIR to the extracted dataset folder.")
        self.bm25 = BM25([c["index_text"] for c in self.chunks])
        self.model_name = embedding_model if embedding_model is not None else config.EMBEDDING_MODEL
        self.encoder, self.matrix = None, None
        if self.model_name and self.model_name.lower() != "none":
            self._build_dense()

    # ---- dense index (optional) -------------------------------------------
    def _prefix(self, kind):
        return f"{kind}: " if "e5" in self.model_name.lower() else ""

    def _build_dense(self):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception:
            print("[kb] sentence-transformers not installed -> BM25-only retrieval")
            return
        try:
            self.encoder = SentenceTransformer(self.model_name)
            key = hashlib.md5((self.model_name + "".join(c["index_text"] for c in self.chunks)).encode()).hexdigest()
            cache = config.CACHE_DIR / f"emb_{key}.npy"
            if cache.exists():
                self.matrix = np.load(cache)
            else:
                texts = [self._prefix("passage") + c["index_text"] for c in self.chunks]
                self.matrix = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.matrix)
        except Exception as exc:
            print(f"[kb] dense index unavailable ({exc}) -> BM25-only retrieval")
            self.encoder = self.matrix = None

    # ---- retrieval ----------------------------------------------------------
    def search(self, query, k=None, include_superseded=False):
        k = k or config.TOP_K
        bm = _minmax(self.bm25.scores(query))
        if self.encoder is not None:
            qv = self.encoder.encode([self._prefix("query") + query], normalize_embeddings=True)[0]
            dense = _minmax([float(x) for x in (self.matrix @ qv)])
            score = [0.6 * d + 0.4 * b for d, b in zip(dense, bm)]
        else:
            score = bm
        ranked = []
        for c, s in zip(self.chunks, score):
            if c["superseded"] and not include_superseded:
                continue
            # small bonus for higher-authority sources (policy > agent guide > help > FAQ > marketing)
            ranked.append((s + 0.03 * (5 - min(c["tier"], 5)), c))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [dict(c, score=round(s, 4)) for s, c in ranked[:k]]

    def dump(self, path):
        Path(path).write_text(json.dumps([{k: v for k, v in c.items() if k != "index_text"} for c in self.chunks],
                                         indent=2, ensure_ascii=False), encoding="utf-8")
