# scripts/hello.py
# Purpose: quick test that your environment and Hugging Face tools work

import time
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate

print("\n🚀 Step 1/4: Loading a small dataset (CNN/DailyMail)...")
ds = load_dataset("cnn_dailymail", "3.0.0", split="validation[:50]")
print(f"✅ Loaded {len(ds)} examples")

print("\n🧠 Step 2/4: Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sample = ds[0]["article"][:2000]
tokens = tok(sample, truncation=True, max_length=512)["input_ids"]
print(f"✅ Tokenized length: {len(tokens)} tokens")

print("\n✂️ Step 3/4: Creating a simple truncation baseline...")
def truncate(text: str, n_tokens: int = 200) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tok.decode(ids)

preds = [truncate(x["article"], 200) for x in ds]
refs = [x["highlights"] for x in ds]

print("\n📊 Step 4/4: Computing ROUGE scores (quality metric)...")
rouge = evaluate.load("rouge")
score = rouge.compute(predictions=preds, references=refs)
print("✅ ROUGE results:", score)
print("\n🎉 Environment works! You’re ready for Day 2.")
