# NLP Finance Project — Research Notebooks

## Overview

Fine-tuning a sentiment analysis model for financial news classification (`bearish`, `neutral`, `bullish`).

**Dataset**: [ArthurMrv/EDGAR-CORPUS-Financial-Summarization-Labeled](https://huggingface.co/datasets/ArthurMrv/EDGAR-CORPUS-Financial-Summarization-Labeled)  
**Final Model**: [ArthurMrv/deberta-v3-ft-financial-news-sentiment-analysis-finetuned](https://huggingface.co/ArthurMrv/deberta-v3-ft-financial-news-sentiment-analysis-finetuned)

---

## Notebook Architecture

### `init_hg.ipynb` *(Initialization — run once)*
Initializes the HuggingFace dataset with empty sentiment columns (`llm_sentiment_class`, `llm_sentiment_rationale`, `llm_sentiment_model`). Only needed for first-time dataset setup.

---

### `update_refined.ipynb` *(Data Labeling)*
Generates sentiment labels using **DeepSeek-V3.2** LLM via HuggingFace Inference API. Processes unlabeled rows in batches, extracts reasoning + score (-2 to +2), and uploads results to the `refined` split.

---

### `fine_tune_model.ipynb` *(Main Pipeline — run this)*
Complete training pipeline:
1. Loads labeled dataset from HuggingFace
2. Evaluates baseline models: `nickmuchi/deberta-v3-base-finetuned-finance-text-classification` and `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis`
3. Fine-tunes the best model (`mrm8488`) with weighted cross-entropy loss
4. Evaluates on held-out test set
5. Uploads fine-tuned model to HuggingFace

**Results**: Weighted F1 = 0.92 (vs baseline macro F1 = 0.60)

---

### `financial_sentiment_analysis.ipynb` *(Alternative/Deprecated)*
Earlier experimentation notebook with similar goals. Not part of the main pipeline.

---

## Pipelines

### Full Training Pipeline
```
1. init_hg.ipynb          → Initialize dataset (once)
2. update_refined.ipynb   → Label data with LLM
3. fine_tune_model.ipynb  → Train & evaluate
```

### Quick Evaluation Only
```
fine_tune_model.ipynb     → Load existing labeled data, evaluate & train
```

---

## Requirements

```bash
pip install transformers datasets torch scikit-learn pandas tqdm wandb
```

Runs on **Google Colab GPU** (T4/A100).
