# scripts/run_baselines.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, torch
import nltk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)

from src.data_utils.mix import build_mixture
from src.data_utils.tokenize import get_tokenizer, truncate_by_tokens
from src.tasks.baselines import bm25_topk
from src.eval.metrics import rouge_scores, exact_match, f1
from src.data_utils.context_filter import compress_to_budget  # BM25-style compressor


# ----------------- simple baselines -----------------
def run_summarization_baselines(dsets=("cnn_dailymail","xsum"), trunc_tokens=200, k_sent=8):
    tok = get_tokenizer("distilbert-base-uncased")
    data = build_mixture(list(dsets), split="validation[:200]")
    data = [ex for ex in data if ex.targets.get("summary")]
    preds_trunc = [truncate_by_tokens(ex.input_text, trunc_tokens, tok) for ex in data]
    preds_bm25  = [bm25_topk(ex.input_text, k_sent=k_sent) for ex in data]
    refs        = [ex.targets["summary"] for ex in data]
    print("\n== Summarization (baselines) ==")
    print("Truncate:", rouge_scores(preds_trunc, refs))
    print("BM25    :", rouge_scores(preds_bm25,  refs))

def run_qa_baseline(dsets=("hotpotqa",), trunc_tokens=400):
    tok = get_tokenizer("distilbert-base-uncased")
    data = build_mixture(list(dsets), split="validation[:200]")
    data = [ex for ex in data if ex.targets.get("answer")]
    preds = [truncate_by_tokens(ex.input_text, trunc_tokens, tok) for ex in data]
    em = sum(exact_match(p, ex.targets["answer"]) for p, ex in zip(preds, data)) / max(1, len(data))
    f1s = sum(f1(p, ex.targets["answer"]) for p, ex in zip(preds, data)) / max(1, len(data))
    print("\n== QA (naive baseline) ==")
    print(f"EM: {em:.3f}  F1: {f1s:.3f}")


# ----------------- helpers -----------------
def _batched_generate(model, tok, texts, max_src_len=512, max_new=80, batch_size=4, prefix=None, device="cpu"):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if prefix:
            batch = [f"{prefix} {t}" for t in batch]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_src_len).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, num_beams=4)
        outs += tok.batch_decode(gen, skip_special_tokens=True)
    return outs

def _maybe_compress_texts_bm25(texts, queries, tok_fast, budget, enable):
    if not enable:
        return texts
    comp = []
    for t, q in zip(texts, queries):
        ct, _ = compress_to_budget(t, q or "", tok_fast, target_tokens=budget)
        comp.append(ct)
    return comp

def _ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass

def _split_sents(text: str):
    _ensure_punkt()
    return nltk.sent_tokenize(text)

def _compress_hybrid(texts, queries, selector_path, budget_tokens,
                     bm25_fallback_ratio=0.3, tok_fast_name="distilbert-base-uncased"):
    """
    Hybrid: selector fills ~70% of the budget; BM25 fills the remaining ~30%.
    """
    from src.tasks.baselines import bm25_topk
    tok_fast = get_tokenizer(tok_fast_name)
    sel_tok = AutoTokenizer.from_pretrained(selector_path)
    sel_model = AutoModelForSequenceClassification.from_pretrained(selector_path).eval()

    compressed = []
    for t, q in zip(texts, queries):
        sents = nltk.sent_tokenize(t)
        if not sents:
            compressed.append(t)
            continue

        # Step 1: selector scores
        with torch.no_grad():
            enc = sel_tok([q]*len(sents), sents, return_tensors="pt",
                          padding=True, truncation=True, max_length=256)
            logits = sel_model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            order = probs.argsort(descending=True).tolist()

        # Step 2: selector fills 70 % of budget
        kept, tokens = [], 0
        sel_budget = int(budget_tokens * (1 - bm25_fallback_ratio))
        for idx in order:
            s = sents[idx]
            tks = len(tok_fast(s, add_special_tokens=False)["input_ids"])
            if tokens and tokens + tks > sel_budget:
                continue
            kept.append(s)
            tokens += tks
            if tokens >= sel_budget:
                break

        # Step 3: BM25 fills remaining 30 %
        remaining_budget = budget_tokens - tokens
        if remaining_budget > 0:
            bm25_text = bm25_topk(t, k_sent=5)
            bm25_sents = nltk.sent_tokenize(bm25_text)
            for s in bm25_sents:
                tks = len(tok_fast(s, add_special_tokens=False)["input_ids"])
                if tokens + tks > budget_tokens:
                    break
                if s not in kept:
                    kept.append(s)
                    tokens += tks

        compressed.append(" ".join(kept))
    return compressed

# ----------------- FLAN-T5 with optional compression -----------------
def run_summarization_llm(dsets=("cnn_dailymail","xsum"), limit=100, batch_size=4,
                          model_name="google/flan-t5-small", compress=False, budget=256, selector_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_llm = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()

    data = build_mixture(list(dsets), split="validation[:200]")
    data = [ex for ex in data if ex.targets.get("summary")][:limit]
    inputs = [ex.input_text for ex in data]
    refs   = [ex.targets["summary"] for ex in data]
    queries = refs  # weak proxy for summarization

    tok_fast = get_tokenizer("distilbert-base-uncased")
    if compress and selector_path:
        inputs_c = _compress_hybrid(inputs, queries, selector_path, budget)
        label = "COMPRESSED (Hybrid: Selector + BM25)"
    elif compress:
        inputs_c = _maybe_compress_texts_bm25(inputs, queries, tok_fast, budget, True)
        label = "COMPRESSED (BM25)"
    else:
        inputs_c = inputs
        label = "FULL"

    preds = _batched_generate(model, tok_llm, inputs_c, max_src_len=512, max_new=80,
                              batch_size=batch_size, prefix="summarize:", device=device)
    print(f"\n== Summarization (FLAN-T5, {label}) ==")
    print(rouge_scores(preds, refs))

def run_qa_llm(dsets=("hotpotqa",), limit=200, batch_size=8,
               model_name="google/flan-t5-small", compress=False, budget=256, selector_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_llm = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()

    data = build_mixture(list(dsets), split="validation[:200]")
    data = [ex for ex in data if ex.targets.get("answer")][:limit]

    questions = [ex.targets["question"] for ex in data]
    contexts  = [ex.input_text for ex in data]

    tok_fast = get_tokenizer("distilbert-base-uncased")
    if compress and selector_path:
        comp_ctx = _compress_with_selector(contexts, questions, selector_path, budget)
        label = "COMPRESSED (Selector)"
    elif compress:
        comp_ctx = _maybe_compress_texts_bm25(contexts, questions, tok_fast, budget, True)
        label = "COMPRESSED (BM25)"
    else:
        comp_ctx = contexts
        label = "FULL"

    prompts = [f"question: {q} context: {c}" for q, c in zip(questions, comp_ctx)]
    preds = _batched_generate(model, tok_llm, prompts, max_src_len=512, max_new=16,
                              batch_size=batch_size, device=device)
    em = sum(exact_match(p, ex.targets["answer"]) for p, ex in zip(preds, data)) / max(1, len(data))
    f1s = sum(f1(p, ex.targets["answer"]) for p, ex in zip(preds, data)) / max(1, len(data))
    print(f"\n== QA (FLAN-T5, {label}) ==")
    print(f"EM: {em:.3f}  F1: {f1s:.3f}")


# ----------------- CLI -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["summ","qa","both"], default="both")
    ap.add_argument("--llm_summ", action="store_true")
    ap.add_argument("--llm_qa", action="store_true")
    ap.add_argument("--compress", action="store_true", help="compress context before LLM")
    ap.add_argument("--budget", type=int, default=256, help="token budget for compression")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--selector_path", type=str, default=None,
                    help="Path to trained selector checkpoint (e.g., checkpoints/selector-distilbert)")
    args = ap.parse_args()

    if args.task in ("summ","both"):
        run_summarization_baselines()
        if args.llm_summ:
            run_summarization_llm(limit=args.limit,
                                  batch_size=args.batch_size,
                                  compress=args.compress,
                                  budget=args.budget,
                                  selector_path=args.selector_path)

    if args.task in ("qa","both"):
        run_qa_baseline()
        if args.llm_qa:
            run_qa_llm(limit=args.limit,
                       batch_size=max(2, args.batch_size),
                       compress=args.compress,
                       budget=args.budget,
                       selector_path=args.selector_path)
