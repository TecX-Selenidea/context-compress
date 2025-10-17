import evaluate
import re

_rouge = evaluate.load("rouge")

def rouge_scores(preds, refs):
    return _rouge.compute(predictions=preds, references=refs)

def exact_match(pred: str, golds):
    golds = [golds] if isinstance(golds, str) else golds
    return int(any(pred.strip().lower() == g.strip().lower() for g in golds))

def f1(pred: str, golds):
    def tok(s): return re.findall(r"\w+", s.lower())
    p = tok(pred)
    best = 0.0
    golds = [golds] if isinstance(golds, str) else golds
    for g in golds:
        g = tok(g)
        common = len(set(p) & set(g))
        if common == 0:
            continue
        prec = common / max(1, len(p))
        rec = common / max(1, len(g))
        best = max(best, 2*prec*rec/(prec+rec))
    return best
