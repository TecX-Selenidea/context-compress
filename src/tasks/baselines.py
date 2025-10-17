import re
import nltk
from rank_bm25 import BM25Okapi

def bm25_topk(text: str, k_sent: int = 8) -> str:
    # ensure punkt is available (no-op if already downloaded)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass

    sents = nltk.sent_tokenize(text)
    if not sents:
        return text
    tokenized = [re.findall(r"\w+", s.lower()) for s in sents]
    bm = BM25Okapi(tokenized)
    # crude “global query”: all words concatenated
    global_query = [w for sent in tokenized for w in sent]
    scores = bm.get_scores(global_query)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_sent]
    idx.sort()
    return " ".join(sents[i] for i in idx)
