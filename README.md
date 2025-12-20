# HPML Project: Low-Latency Limit Order Book Prediction

## Team Information

- **Members**:
- Max Schettewi (mjs2430)
- Shubham Mangalvedhe (srm2251)
- William Wadolowski (wrw2113)
- Nikhil Gahlot (ng2990)

---

## 1. Problem Statement

High-frequency trading systems need to predict short-term price movements from limit order book (LOB) data, but any predictive edge is only useful if it can be produced within very tight latency budgets. The core problem is a latency–accuracy tradeoff: models that capture richer temporal structure (e.g., Transformers) can improve prediction quality, but they are often too slow for deployment in latency-critical settings, while faster CNN-style models may miss longer-range dependencies and underperform on accuracy.

In this project, we frame the task as 3-class mid-price direction prediction (down / stationary / up) on the FI-2010 benchmark at a short horizon (k=10), and we evaluate both predictive performance (accuracy, macro-F1) and deployment performance (latency and throughput). Our goal is to understand when a Transformer’s accuracy gains justify its compute cost and which system-level optimizations can narrow the gap.

---

## 2. Model Description

We implement and compare two architectures for short-horizon mid-price movement prediction on the FI-2010 limit order book benchmark. Both models take the same supervised input: a rolling window of T = 100 events, where each event is represented by 144 LOB features (so each sample is a 100 × 144 sequence). The prediction target is a 3-class direction label at horizon k = 10 (down / stationary / up).

CNN Baseline (DeepLOB-style):
The CNN baseline treats each input window as a 2D feature map by reshaping the sequence to [B, 1, 144, 100] (features as “height”, time as “width”). It uses three Conv2D blocks with increasing channel sizes 32 → 64 → 128, with pooling along the time dimension to reduce temporal resolution. A global average pooling layer aggregates the learned representation, followed by a small fully connected head that outputs 3-class logits. This model is designed as a strong low-latency baseline.

Transformer Classifier:
The Transformer model linearly projects the 144 input features into an embedding space with d_model = 128, adds sinusoidal positional encoding to preserve event ordering, and passes the sequence through a 4-layer Transformer encoder with 8 attention heads and d_ff = 512. We aggregate the sequence using mean pooling over time, then apply an MLP head (128 → 64 → 3) to produce logits. This model is intended to better capture longer-range temporal dependencies at the cost of higher compute.

Training Notes:
Both models are trained as multi-class classifiers using class-weighted cross entropy to mitigate imbalance (the “stationary” class dominates), and optimized with AdamW.

---

## 3. Final Results Summary

We benchmark a Transformer encoder against a DeepLOB-style CNN baseline on FI-2010 (NoAuction_Zscore) for 3-class mid-price direction prediction at horizon k=10, using T=100 event windows and 144 features per event.

Predictive Performance (Held-out Test, k=10)
DeepLOB (CNN): 63.02% accuracy, 0.58 macro-F1, 0.42 precision (Up/Down)
Transformer (Ours): 67.91% accuracy, 0.65 macro-F1, 0.49 precision (Up/Down)
Overall, the Transformer improves accuracy by +4.9% and macro-F1 by +0.07, with stronger minority-class precision.

Inference Benchmarks (NVIDIA A100, 100 runs)
HFT / single-sample (Batch=1):
CNN: 0.69 ms latency ( 1,458 samp/s )
Transformer: 1.57 ms latency ( 637 samp/s )
The CNN is about 2.2× faster in pure latency.

Training/backtest throughput setting (Batch=256):
CNN: 10.60 ms per batch ( 24,146 samp/s )
Transformer: 6.58 ms per batch ( 38,933 samp/s )
The Transformer is about 61% higher throughput at scale.

Quantization Finding (Dynamic INT8):
Dynamic quantization reduced model size slightly (3.35 MB → 3.33 MB) but did not improve latency in the latency-critical regime; it regressed inference from 1.57 ms → 1.65 ms (≈ 0.95×). This highlights that compression is not automatically speed: overhead can dominate for small-batch, latency-focused workloads.

Key Takeaway:
The Transformer delivers better signal quality (accuracy/F1), but you “pay” for it with ~+0.88 ms additional batch=1 latency. The CNN is preferred for ultra-low-latency execution, while the Transformer becomes attractive when predictive power matters and/or when inference can be batched for throughput.

---

## 4. Reproducibility Instructions

### A. Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### B. Wandb Dashboard
View training and evaluation metrics [here](https://wandb.ai/mjs2430-columbia-university/HPML%20LOB?nw=nwusermjs2430). 

Since the Wandb project is by invite only, it is also shared Google Drive (Columbia University emails) [here](https://drive.google.com/drive/folders/1xVU8LKgV-kNAkewWp0iTdvz1tReQjOy9).

---

### C. Specify for Training or For Inference or if Both

- **To download and prepare the dataset:**
It will prompt for a dataset location since the dataset link is one time use. Go to the provided link, click the 3 dots
and paste the url only for wget.

```bash
python scripts/load_data.py
```

- **To train the model from scratch:**
It will prompt for a wandb API key, you may opt out of wandb logging or provide a key as required.

```bash
python scripts/train.py 
```

- **To evaluate the model:**

```bash
python scripts/evaluate.py
```

- **To benchmark quantization:**

```bash
python scripts/benchmark_quantized.py
```

- **To profile the model:**

```bash
python scripts/profile_model.py
```

---

### D. Evaluation

To evaluate the trained model:

```bash
python scripts/evaluate.py
```

---

### E. Quickstart: Minimum Reproducible Result

To reproduce our minimum reported result (e.g., XX.XX% accuracy), run:

```bash
# Step 1: Set up environment
pip install -r requirements.txt
# Step 2: Download dataset
python scripts/load_data.py
# Step 3: Run training (or skip if checkpoint is provided)
python scripts/train.py 
# Step 4: Evaluate
python scripts/evaluate.py
```

---

## 5. Notes

- All entry points are located in `scripts/`
- All source code is located in `src/`
- Model checkpoints are saved to `checkpoints/`
- Data is stored in `data/`
- Logs and visualizations are saved to `logs/`
- Visualization notebook is saved in `notebooks/`
