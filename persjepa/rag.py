"""BM25 retrieval-augmented prompting baseline (LaMP protocol, Salemi et al.).

History for a user = (a) the LaMP raw profile items stored in metadata.raw_profile
when present, else (b) the user's own calibration records (prompt -> target
pairs), i.e. exactly the information the routed SAE's z_u is built from.
Top-k items by BM25 against the query are placed in the prompt as examples.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from persjepa.routing import group_of

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall(text.lower())


def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    q_tokens = tokenize(query)
    doc_tokens = [tokenize(d) for d in docs]
    n = len(docs)
    avgdl = sum(len(d) for d in doc_tokens) / max(n, 1)
    df: Counter = Counter()
    for toks in doc_tokens:
        df.update(set(toks))
    scores = []
    for toks in doc_tokens:
        tf = Counter(toks)
        score = 0.0
        for term in q_tokens:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * len(toks) / max(avgdl, 1))
            score += idf * tf[term] * (k1 + 1) / denom
        scores.append(score)
    return sorted(range(n), key=lambda i: -scores[i])


def item_text(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        parts = [str(item[k]) for k in ("title", "text", "abstract", "review", "description") if item.get(k)]
        return " | ".join(parts) if parts else " ".join(str(v) for v in item.values() if isinstance(v, str))
    return str(item)


class HistoryIndex:
    """user -> list of (search text, display text)."""

    def __init__(self) -> None:
        self.items: dict[str, list[tuple[str, str]]] = defaultdict(list)

    @classmethod
    def from_calibration(cls, records, group_field=None, *, prompt_field="prompt", target_field="target", max_chars=400) -> "HistoryIndex":
        idx = cls()
        for r in records:
            u = group_of(r, group_field)
            p, t = str(r.get(prompt_field, "")).strip(), str(r.get(target_field, "")).strip()
            idx.items[u].append((p + " " + t, f"Request: {p[:max_chars]}\nUser's response: {t[:max_chars]}"))
        return idx

    def retrieve(self, user: str, query: str, top_k: int = 5, *, exclude_text: str | None = None) -> list[str]:
        hist = self.items.get(user, [])
        hist = [h for h in hist if exclude_text is None or exclude_text not in h[0]]
        if not hist:
            return []
        order = bm25_rank(query, [h[0] for h in hist])[:top_k]
        return [hist[i][1] for i in order]


def rag_prompt(query: str, examples: list[str], profile_header: str | None = None) -> str:
    if not examples and not profile_header:
        return query
    lines = []
    if profile_header:
        lines.append(profile_header.strip())
    if examples:
        lines.append("Examples of this user's past responses:")
        lines += [f"{i}. {e}" for i, e in enumerate(examples, 1)]
    lines.append(f"\nRequest: {query.strip()}")
    lines.append("User's response:")
    return "\n".join(lines)


def raw_profile_examples(metadata: dict | None, query: str, top_k: int = 5) -> list[str] | None:
    """LaMP-style: metadata.raw_profile items ranked by BM25. None when absent."""
    items = (metadata or {}).get("raw_profile")
    if not items:
        return None
    texts = [item_text(it) for it in items]
    order = bm25_rank(query, texts)[:top_k]
    return [texts[i][:400] for i in order]
