# src/data_utils/context_filter.py
"""
Simple, fast context compressor.
- Splits text into sentences.
- Scores sentences vs. a query with BM25.
- Returns the top sentences that fit a token budget.
"""
from typing import List, Tuple
import re, nltk
from rank_bm25 import BM25Okapi

def _ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass

def split_sentences(text: str) -> List[str]:
    _ensure_punkt()
    # keep only “readable” whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return nltk.sent_tokenize(text)

def estimate_tokens(s: str, tok) -> int:
    # fast token count using your HF tokenizer
    return len(tok(s, add_special_tokens=False)["input_ids"])

def compress_to_budget(text: str, query: str, tok, target_tokens: int = 256) -> Tuple[str, List[int]]:
    """
    Selects BM25-top sentences until we reach ~target_tokens.
    Returns (compressed_text, kept_indices).
    """
    sents = split_sentences(text)
    if not sents:
        return text, []

    # tokenize sentences for BM25 (very light)
    tokenized = [re.findall(r"\w+", s.lower()) for s in sents]
    bm = BM25Okapi(tokenized)

    # query → tokens; if empty, fall back to first 20 words of text
    q = query.strip() or " ".join(re.findall(r"\w+", text.lower())[:20])
    q_tokens = re.findall(r"\w+", q.lower())
    scores = bm.get_scores(q_tokens)

    # rank sentences by score (desc)
    order = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)

    kept, total_tokens = [], 0
    for i in order:
        t = estimate_tokens(sents[i], tok)
        if total_tokens + t > target_tokens and kept:
            continue
        kept.append(i)
        total_tokens += t
        if total_tokens >= target_tokens:
            break

    kept.sort()  # preserve original order for readability
    comp = " ".join(sents[i] for i in kept) if kept else sents[0]
    return comp, kept
