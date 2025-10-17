# Context Compression Pipeline

## Status (Day 10)
- Baselines: Truncate / BM25 / FLAN-T5 working end-to-end
- Selector: DistilBERT trained (CNN/DailyMail, 5k, 2 epochs, ROUGE-L recall supervision)
- Compression: BM25, Selector, Hybrid (Selector + BM25)
- Best (Summ): Selector ~0.27 ROUGE-1 @ 384 tokens; Hybrid consistently > BM25

## Quick start
**Run summarization (Selector compression):**
```bash
python -m scripts.run_baselines --task summ --llm_summ --compress --budget 384 --selector_path checkpoints/selector-distilbert --limit 50

Train Selector:
python -m scripts.train_selector --mode summ --train "train[:5000]" --val "validation[:1200]" --epochs 2 --bs 16 --max_len 256
