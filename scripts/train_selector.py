# scripts/train_selector.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from inspect import signature
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

from src.data_utils.selector_dataset import SelectorDataset


class RawSelDataset(torch.utils.data.Dataset):
    """Simple wrapper so we can collate with a tokenizer."""
    def __init__(self, sel_ds: SelectorDataset):
        self.data = sel_ds.examples
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        ex = self.data[i]
        return {"query": ex.query, "sentence": ex.sentence, "labels": int(ex.label)}


def build_training_args(args) -> TrainingArguments:
    """Version-agnostic TrainingArguments (v4 uses evaluation_strategy, v5 uses eval_strategy)."""
    base = dict(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],  # works on v4/v5 to disable W&B etc.
    )
    sig = signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in sig:
        base["evaluation_strategy"] = "epoch"
    else:
        base["eval_strategy"] = "epoch"
    return TrainingArguments(**base)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["summ", "qa"], default="summ",
                    help="summarization supervision (CNN/DM) or QA supervision (HotpotQA)")
    ap.add_argument("--train", type=str, default="train[:3000]",
                    help='HF split for training slice, e.g. "train[:3000]"')
    ap.add_argument("--val", type=str, default="validation[:800]",
                    help='HF split for validation slice, e.g. "validation[:800]"')
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--out", type=str, default="checkpoints/selector-distilbert")
    args = ap.parse_args()

    # Build weakly supervised selector datasets
    sel_train = SelectorDataset(mode=args.mode, split=args.train, k_pos=5)
    sel_val   = SelectorDataset(mode=args.mode, split=args.val,   k_pos=5)
    print(f"[data] mode={args.mode}  train={len(sel_train)}  val={len(sel_val)}")

    d_train = RawSelDataset(sel_train)
    d_val   = RawSelDataset(sel_val)

    tok = AutoTokenizer.from_pretrained(args.model_name)

    def collate(batch):
    # Accept dicts, SelectorExample objects, or (query, sentence, label) tuples
        if hasattr(batch[0], "query"):  # SelectorExample objects
            q = [b.query for b in batch]
            s = [b.sentence for b in batch]
            y = [int(b.label) for b in batch]
        elif isinstance(batch[0], dict):  # dicts from RawSelDataset
            q = [b.get("query", "") for b in batch]
            s = [b.get("sentence", "") for b in batch]
            # accept 'labels' or 'label'
            y = [int(b.get("labels", b.get("label"))) for b in batch]
        else:  # tuples
            q = [b[0] for b in batch]
            s = [b[1] for b in batch]
            y = [int(b[2]) for b in batch]

        enc = tok(q, s, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        enc["labels"] = torch.tensor(y, dtype=torch.long)
        return enc

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    training_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=d_train,
        eval_dataset=d_val,
        data_collator=collate,
        compute_metrics=compute_metrics,
        tokenizer=tok,
    )

    trainer.train()
    print("[eval]", trainer.evaluate())

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
