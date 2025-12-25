# NLP Finance Project — Model Comparison, Fine-tuning & Knowledge Distillation

## Overview
This project aims to **benchmark and improve financial text-classification models** on multi-source financial news.  
We will:
1. **Generate or standardize labels** for a financial news dataset (optionally using LLMs),
2. **Evaluate** two pretrained transformer models,
3. **Select the best model** and **fine-tune** it on the labeled dataset,
4. **Distill** the fine-tuned model (teacher) into a smaller model (student),
5. Compare **baseline vs fine-tuned vs distilled** models.


It should be a good addition to : <https://github.com/ArthurrMrv/graph_project.git> Enabeling this project to be based on a newer, more efficient and optimised sentiment analysis model.
---

## Objectives
- Compare pretrained finance-oriented classifiers on a shared dataset.
- Build a reproducible labeling + training + evaluation pipeline.
- Fine-tune the best-performing model.
- Apply **knowledge distillation** to obtain a lighter model with comparable accuracy.
- Provide a clear report of performance, trade-offs (speed/size vs quality), and error analysis.

---

## Dataset
We use the Hugging Face dataset:

- **financial-news-multisource**  
  <https://huggingface.co/datasets/Brianferrell787/financial-news-multisource>

**Key idea:** labels are missing, we will generate normalized labels (`{bearish, neutral, bullish}`).

---

## Models Compared (Baselines)
We evaluate two pretrained text-classification pipelines:

```python
from transformers import pipeline

pipes = {
  "nickmuchi": pipeline(
      "text-classification",
      model="nickmuchi/deberta-v3-base-finetuned-finance-text-classification"
  ),
  "mrm8488": pipeline(
      "text-classification",
      model="mrm8488/deberta-v3-ft-financial-news-sentiment-analysis"
  )
}
```

---

## Labeling Strategy

The dataset does not include usable labels, we generate labels using:

* API calls (e.g., **GPT-5 mini**, **Gemini-3 Flash**),
* or load an LLM model to generate labels (e.g., **Mistral** / **Llama** endpoints) depending on compute + budget constraints.

### Label taxonomy (default)

* `bearish`
* `neutral`
* `bullish`

### Quality control

* Keep a small manually-verified subset (gold set) to estimate noise.
* Cache LLM outputs to avoid repeated costs.

---

## Methodology

### Step 1 — Data Preparation

* Load dataset with `datasets`
* Basic cleaning:

  * remove duplicates (if needed),
  * filter extremely short/empty texts,
  * unify text fields.
* Create `train / val / test` split (example 70/15/15), with a fixed seed.

### Step 2 — Baseline Evaluation

Evaluate both baseline models on the labeled validation/test set:

* Accuracy
* Macro F1
* Precision/Recall per class
* Confusion matrix

### Step 3 — Teacher Selection + Fine-tuning

Pick the best baseline model and fine-tune on `train`:

* Use Hugging Face `Trainer`
* Track metrics on `val`
* Save best checkpoint

### Step 4 — Knowledge Distillation

Distill the fine-tuned teacher into a smaller student model:

* Distillation loss: KL divergence on logits (teacher vs student)
* Optional mixed objective: `alpha * KL + (1-alpha) * CE(labels)`
* Goal: near-teacher performance with better speed/size

### Step 5 — Final Comparison

Compare 2 variants:

1. **Pretrained baseline**
2. **Distilled student**

---

## Evaluation Protocol

### Metrics

* **Accuracy**
* **Macro F1** (robust under class imbalance)
* **Precision / Recall** per class
* Confusion matrix
* Optional: inference latency + model size

### Experimental Setup

* Fixed random seed
* Same train/val/test split
* Same preprocessing for all models
* Report confidence via multiple runs if time allows

---

## Engineering & Reproducibility

### Tech Stack

* Python
* Hugging Face: `transformers`, `datasets`, `evaluate`
* PyTorch
* (Optional) Weights & Biases for experiment tracking
* Google Colab (GPU)

---

## Installation

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2) Install dependencies

```bash
pip install -U pip
pip install transformers datasets evaluate torch scikit-learn pandas numpy tqdm
```

Optional:

```bash
pip install wandb
```

---

## Usage (Planned)

### 1) Download + preprocess dataset

```bash
python data/preprocess.py
python data/make_splits.py
```

### 2) (Optional) Generate labels with LLM

```bash
python data/labeling_llm.py --model mistral --cache_dir .cache/labels
```

### 3) Evaluate baselines

```bash
python evaluation/evaluate_baselines.py
```

### 4) Fine-tune best teacher

```bash
python models/finetune_teacher.py --config configs/finetune.yaml
```

### 5) Distill to student

```bash
python models/distill_student.py --config configs/distill.yaml
```

### 6) Compare all

```bash
python evaluation/compare_all.py
```

---

## Expected Results

* Fine-tuning should improve macro-F1 compared to the pretrained baseline.
* Distillation should retain most of the teacher’s performance while:
  * reducing model size,
  * improving inference speed.

---

## Deliverables

* Reproducible training + evaluation pipeline
* Model checkpoints (teacher + student)
* Final report:

  * baseline comparison,
  * fine-tuning results,
  * distillation results,
  * error analysis,
  * compute/cost discussion.

---

## Notes on Compute

University compute resources may be restricted. This project is designed to run on:

* **Google Colab GPU** (T4/A100 depending on availability)
* Small-to-medium transformer models and efficient training settings (batch size, gradient accumulation).

---

## License

Specify your license here (e.g., MIT).

---

## Acknowledgements

* Hugging Face Transformers & Datasets
* Dataset authors: `Brianferrell787/financial-news-multisource`
