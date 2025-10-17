# src/data_utils/selector_dataset.py
"""
Sentence-selection dataset for training a learnable compressor.
- For summarization: query = reference summary; positives = sentences most similar to the summary.
- For QA: query = question; positives = sentences containing (or highly similar to) the answer.
Returns (query, sentence, label) pairs; tokenization happens in the collate.
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import re, nltk
from datasets import load_dataset
from rouge_score import rouge_scorer

def _ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        try: nltk.download("punkt_tab")
        except Exception: pass

def _sentences(text: str) -> List[str]:
    _ensure_punkt()
    text = re.sub(r"\s+", " ", text).strip()
    return nltk.sent_tokenize(text)

def _topk_by_rougeL(sentences: List[str], query: str, k: int = 7, use_recall: bool = True) -> List[int]:
    """
    Rank sentences by ROUGE-L against the query.
    use_recall=True favors coverage (better for summarization),
    otherwise uses F1 (balanced).
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scored = []
    for i, s in enumerate(sentences):
        r = scorer.score(query, s)["rougeL"]
        val = r.recall if use_recall else r.fmeasure
        scored.append((val, i))
    scored.sort(reverse=True)
    k = max(1, min(k, len(sentences)))
    return [i for _, i in scored[:k]]


class SelectorExample:
    __slots__ = ("query","sentence","label")
    def __init__(self, query: str, sentence: str, label: int):
        self.query = query; self.sentence = sentence; self.label = int(label)

class SelectorDataset:
    """
    mode='summ' -> CNN/DM (default) using reference summary as query (weak supervision)
    mode='qa'   -> HotpotQA using question as query; mark sentences containing the answer as positives
    """
    def __init__(self, mode: str = "summ", split: str = "train[:2000]", k_pos: int = 5):
        assert mode in ("summ","qa")
        self.mode = mode
        self.examples: List[SelectorExample] = []

        if mode == "summ":
            ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
            for r in ds:
                doc = r["article"]; query = r["highlights"] or ""
                sents = _sentences(doc)
                pos_idx = _topk_by_rougeL(sents, query, k=k_pos)
                pos_set = set(pos_idx)
                # add positives and a matched number of negatives
                for i in pos_idx:
                    self.examples.append(SelectorExample(query, sents[i], 1))
                neg_added = 0
                for i, s in enumerate(sents):
                    if i in pos_set: continue
                    self.examples.append(SelectorExample(query, s, 0))
                    neg_added += 1
                    if neg_added >= len(pos_idx)*2:  # 2:1 neg:pos
                        break

        else:  # QA mode
            ds = load_dataset("hotpot_qa", "distractor", split=split)
            for r in ds:
                query = r.get("question","")
                answer = (r.get("answer","") or "").strip().lower()
                # hotpot 'context' is a list of [title, sentences[]]
                ctx = []
                for title, sents in r.get("context", []):
                    ctx.extend(sents)
                if not ctx:
                    continue
                sents = ctx
                pos_idx = []
                for i, s in enumerate(sents):
                    if answer and answer in s.lower():
                        pos_idx.append(i)
                # fallback: top-k by rougeL if no exact match
                if not pos_idx and answer:
                    pos_idx = _topk_by_rougeL(sents, answer, k=k_pos)
                pos_set = set(pos_idx)
                for i in pos_idx:
                    self.examples.append(SelectorExample(query, sents[i], 1))
                neg_added = 0
                for i, s in enumerate(sents):
                    if i in pos_set: continue
                    self.examples.append(SelectorExample(query, s, 0))
                    neg_added += 1
                    if neg_added >= max(1, len(pos_idx))*2:
                        break

    def __len__(self): return len(self.examples)
    def __getitem__(self, i: int) -> SelectorExample: return self.examples[i]

def selector_collate_fn(batch: List[SelectorExample], tokenizer, max_len: int = 256):
    queries  = [b.query for b in batch]
    sentences= [b.sentence for b in batch]
    labels   = [b.label for b in batch]
    enc = tokenizer(queries, sentences, return_tensors="pt",
                    truncation=True, padding=True, max_length=max_len)
    enc["labels"] = __import__("torch").tensor(labels, dtype=__import__("torch").long)
    return enc
