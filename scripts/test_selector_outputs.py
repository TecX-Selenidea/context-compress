# scripts/test_selector_outputs.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse, torch, nltk, re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

from src.tasks.baselines import bm25_topk
from src.data_utils.tokenize import get_tokenizer


def _ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


def sent_split(text: str):
    _ensure_punkt()
    text = re.sub(r"\s+", " ", text).strip()
    return nltk.sent_tokenize(text)


def selector_rank(query: str, sents, selector_path: str, max_len=256):
    tok = AutoTokenizer.from_pretrained(selector_path)
    model = AutoModelForSequenceClassification.from_pretrained(selector_path).eval()
    with torch.no_grad():
        enc = tok([query]*len(sents), sents, return_tensors="pt",
                  padding=True, truncation=True, max_length=max_len)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]   # P(keep)
    order = probs.argsort(descending=True).tolist()
    return order, probs.tolist()


def compress_with_selector(query: str, text: str, selector_path: str, token_budget=256,
                           fast_tok_name="distilbert-base-uncased"):
    sents = sent_split(text)
    if not sents: 
        return text, []
    order, probs = selector_rank(query, sents, selector_path)
    fast_tok = get_tokenizer(fast_tok_name)

    kept, tok_sum = [], 0
    for idx in order:
        s = sents[idx]
        n = len(fast_tok(s, add_special_tokens=False)["input_ids"])
        if tok_sum and tok_sum + n > token_budget:
            continue
        kept.append((idx, s))
        tok_sum += n
        if tok_sum >= token_budget:
            break

    kept.sort(key=lambda x: x[0])  # original order
    comp = " ".join(s for _, s in kept) if kept else sents[0]
    return comp, [(i, probs[i]) for i, _ in kept]


def main():
    ap = argparse.ArgumentParser(description="Visualize Selector vs BM25 on one example.")
    ap.add_argument("--selector_path", type=str, default="checkpoints/selector-distilbert",
                    help="Path to trained selector checkpoint folder")
    ap.add_argument("--use_dataset", action="store_true",
                    help="If set, pull one CNN/DailyMail example instead of --text")
    ap.add_argument("--index", type=int, default=0, help="Example index when using dataset")
    ap.add_argument("--text", type=str, default="",
                    help="Custom text to analyze (ignored if --use_dataset)")
    ap.add_argument("--query", type=str, default="", 
                    help="Custom query. If empty with dataset, uses reference summary.")
    ap.add_argument("--k", type=int, default=5, help="How many top sentences to display")
    ap.add_argument("--budget", type=int, default=256,
                    help="Optional token budget for selector compression (shows compressed text)")
    args = ap.parse_args()

    # 1) Pick text + query
    if args.use_dataset:
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="validation")
        ex = ds[int(args.index)]
        text = ex["article"]
        query = args.query or ex["highlights"] or "summarize the article"
        print(f"[dataset] CNN/DailyMail validation[{args.index}] | query is summary")
    else:
        if not args.text:
            print("Please pass --text 'your document ...' or use --use_dataset")
            return
        text = args.text
        query = args.query or "summarize the document"

    # 2) Sentence list
    sents = sent_split(text)
    print(f"\nTotal sentences: {len(sents)}")

    # 3) Selector ranking
    sel_order, sel_probs = selector_rank(query, sents, args.selector_path)
    print("\n=== Selector: top sentences ===")
    for i, idx in enumerate(sel_order[:args.k]):
        print(f"[{i+1:02d}] p_keep={sel_probs[idx]:.3f}  {sents[idx]}")

    # 4) BM25 selection (k sentences)
    print("\n=== BM25: top sentences (k) ===")
    print(bm25_topk(text, k_sent=args.k))

    # 5) Selector compression to a token budget
    comp, kept = compress_with_selector(query, text, args.selector_path, token_budget=args.budget)
    print(f"\n=== Selector compression @ {args.budget} tokens ===")
    print(comp)
    if kept:
        kept_str = ", ".join([f"#{i}(p={p:.2f})" for i, p in kept])
        print(f"(kept indices & probs): {kept_str}")


if __name__ == "__main__":
    main()
