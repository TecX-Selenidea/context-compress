from datasets import load_dataset
from .schema import Example

# ---- Summarization: CNN/DailyMail ----
def map_cnn_dailymail(split: str = "validation[:200]"):
    ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
    def to_ex(r):
        return Example(
            source_id=str(r["id"]),
            input_text=r["article"],
            targets={"summary": r["highlights"], "question": None, "answer": None},
            aux={"title": r.get("title",""), "dataset": "cnn_dailymail"}
        )
    return [to_ex(r) for r in ds]

# ---- Summarization: XSum ----
def map_xsum(split: str = "validation[:200]"):
    ds = load_dataset("EdinburghNLP/xsum", split=split)
    def to_ex(r):
        return Example(
            source_id=str(r.get("id", "")),
            input_text=r["document"],
            targets={"summary": r["summary"], "question": None, "answer": None},
            aux={"title": r.get("title",""), "dataset": "xsum"}
        )
    return [to_ex(r) for r in ds]

# ---- QA: HotpotQA (simple mapping; question text only for now) ----
def map_hotpotqa(split: str = "validation[:200]"):
    ds = load_dataset("hotpot_qa", "distractor", split=split)
    def to_ex(r):
        answer = r.get("answer", "")
        return Example(
            source_id=str(r.get("_id","")),
            input_text=r.get("question",""),
            targets={"summary": None, "question": r.get("question",""), "answer": answer},
            aux={"dataset": "hotpotqa"}
        )
    return [to_ex(r) for r in ds]
