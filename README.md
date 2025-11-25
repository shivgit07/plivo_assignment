# plivo_assignment

Here is the complete content for your `README.md` file in a single code block.

# PII Entity Recognition for Noisy STT Transcripts

## 📌 Project Overview
This project implements a high-performance Named Entity Recognition (NER) system designed to identify Personally Identifiable Information (PII) in noisy Speech-to-Text (STT) transcripts.

The model detects the following entities:
* **PII:** `CREDIT_CARD`, `PHONE`, `EMAIL`, `PERSON_NAME`, `DATE`
* **Non-PII:** `CITY`, `LOCATION`

The system is optimized for **low-latency CPU inference** (targeting <20ms p95) while maintaining high precision for sensitive PII data.

## 🚀 Key Technical Approach

To achieve the balance between speed and accuracy on a small, noisy dataset, the following architectural changes were made to the baseline:

### 1. Model Selection: DistilRoBERTa
We switched from the standard `distilbert-base-uncased` to **`distilroberta-base`**. RoBERTa is pretrained on a larger corpus with a more robust masking procedure, allowing it to handle "noisy" text (missing punctuation, spoken numbers, grammatical errors) significantly better than standard BERT.

### 2. Latency Optimization
* **Sequence Truncation:** The `max_length` was optimized to **128 tokens**. Since STT utterances are typically short, reducing sequence length provided a quadratic reduction in attention mechanism complexity, resulting in a **p95 latency of ~7.2ms** on CPU.

### 3. Combatting Overfitting on Small Data
Given the limited training data (~850 examples), two specific techniques were implemented to improve generalization:
* **Layer Freezing:** We implemented dynamic layer freezing (freezing embeddings + bottom 3 transformer layers). This forces the model to rely on pre-trained linguistic features rather than memorizing the small training set.
* **Weighted Cross-Entropy Loss:** We utilized Inverse Frequency Weighting during training. This assigns higher penalties to errors made on rare classes (like `EMAIL` or `CREDIT_CARD`), forcing the model to prioritize recall for these critical entities.

## 📊 Performance Metrics

### Latency (CPU)
The model significantly exceeds the latency requirement of 20ms.
```text
Latency over 100 runs (batch_size=1):
  p50: 5.34 ms
  p95: 7.20 ms
````

### Dev Set Evaluation

The model achieves high performance on the standard development set.

```text
Per-entity metrics:
CITY            P=0.966 R=1.000 F1=0.982
CREDIT_CARD     P=1.000 R=1.000 F1=1.000
DATE            P=1.000 R=1.000 F1=1.000
EMAIL           P=0.955 R=0.977 F1=0.966
LOCATION        P=1.000 R=1.000 F1=1.000
PERSON_NAME     P=1.000 R=1.000 F1=1.000
PHONE           P=1.000 R=1.000 F1=1.000

Macro-F1: 0.993

PII-only metrics: P=0.993 R=0.997 F1=0.995
Non-PII metrics: P=0.978 R=1.000 F1=0.989
```

### Stress Test Evaluation (Robustness)

The stress set contains adversarial examples (e.g., spoken emails like "dot com", spelled out numbers).

```text
Per-entity metrics:
CITY            P=0.857 R=0.750 F1=0.800
CREDIT_CARD     P=0.277 R=0.450 F1=0.343
DATE            P=0.952 R=1.000 F1=0.976
EMAIL           P=0.000 R=0.000 F1=0.000
PERSON_NAME     P=0.262 R=0.975 F1=0.413
PHONE           P=0.257 R=0.450 F1=0.327

Macro-F1: 0.476

PII-only metrics: P=0.368 R=0.730 F1=0.489
Non-PII metrics: P=0.857 R=0.750 F1=0.800
```

## 🛠️ Installation & Usage

1.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model**

    ```bash
    python src/train.py \
      --model_name distilroberta-base \
      --train data/train.jsonl \
      --dev data/dev.jsonl \
      --out_dir out \
      --epochs 10 \
      --batch_size 16 \
      --freeze_layers 3 \
      --max_length 128
    ```

3.  **Run Inference**

    ```bash
    python src/predict.py \
      --model_dir out \
      --input data/test.jsonl \
      --output out/predictions.json \
      --max_length 128
    ```

## 📂 Project Structure

  * `src/train.py`: Main training loop with weighted loss and layer freezing logic.
  * `src/model.py`: Model definition using Hugging Face AutoModel.
  * `src/dataset.py`: Data loader converting JSONL to tokenized BIO tags.
  * `src/predict.py`: Inference script for generating span predictions.
  * `src/eval_span_f1.py`: Metric evaluation script.
  * `src/measure_latency.py`: Latency benchmarking script.
  * `data/`: Contains train, dev, and stress datasets.

<!-- end list -->

```
```
