from transformers import AutoTokenizer

def get_tokenizer(name: str = "distilbert-base-uncased"):
    return AutoTokenizer.from_pretrained(name, use_fast=True)

def truncate_by_tokens(text: str, n: int, tok) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids)
