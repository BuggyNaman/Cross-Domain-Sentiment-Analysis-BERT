# Cross-Domain Sentiment Analysis via Disentangled Token-Level Attention & Confidence-Gated Prototypical Learning

An enhanced PyTorch + BERT framework for **Unsupervised Cross-Domain Sentiment Analysis (CDSA)** across Amazon product domains (*Books, DVDs, Electronics, Kitchen*), paired with an interactive **Streamlit** explainability dashboard.

---

## Key Architectural Enhancements

1. **Dual Token-Level Attention (`TokenAttention`)**: Replaces static `[CLS]` pooling with separate learned attention heads for **Sentiment** (domain-invariant opinion cues) and **Domain** (product-specific context).
2. **Active Orthogonal Disentanglement Loss**: Combines KL divergence with a squared cosine similarity penalty between sentiment and domain representations to enforce feature separation.
3. **Confidence-Gated Prototypical Alignment**: Anchors target-domain prototype alignment to high-confidence pseudo-labels ($\tau \ge 0.85$) activated from Epoch 2 onward (scaled by $0.1$), preventing early mode collapse on distant domain shifts such as `DVDs -> Electronics`.

---

## Benchmark Results (Amazon Reviews)

Evaluated under controlled settings (`bert-base-uncased`, `max_seq_length=256`, `labeled_num=1600`, `epochs=4`, `seed=42`):

| Transfer Task | BERT Baseline (Replicated) | Proposed (Attention + Ortho + Gate) | Improvement (Δ) |
| :--- | :---: | :---: | :---: |
| **Books ➔ DVDs** | 86.10% | **89.40%** | **+3.30%** |
| **DVDs ➔ Electronics** | 87.85% | **89.20%** | **+1.35%** |
| **Electronics ➔ Kitchen** | 92.25% | **93.75%** | **+1.50%** |
| **Kitchen ➔ Books** | 81.05% | **84.25%** | **+3.20%** |
| **Average** | **86.81%** | **89.15%** | **+2.34%** |

---

## Interactive Streamlit Dashboard

The included `app.py` provides:
- Real-time sentiment classification with confidence scores across all 4 transfer pairs.
- **Disentangled Word-Level Attention Heatmaps** highlighting sentiment-driving vs. domain-specific words.
- Top-5 token attribution bar charts and benchmark comparison views.

<!-- Uncomment after uploading screenshots to assets/
![Dashboard Demo](assets/dashboard_demo.png)
![Benchmarks](assets/benchmarks.png)
-->

---

## Repository Structure

```text
.
├── app.py                        # Interactive Streamlit explainability dashboard
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes heavy .pt weights & raw data
├── assets/                       # Dashboard screenshots
└── src/
    ├── run_base_sa.py            # Phase 1: BERT baseline & pseudo-label generation
    └── run_prototype_attention.py # Phase 2: Disentangled Attention + Prototypical model
```

---

## Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/<your-username>/Cross-Domain-Sentiment-Analysis-BERT.git
cd Cross-Domain-Sentiment-Analysis-BERT
pip install -r requirements.txt
```

### 2. Run Training (Example: Electronics ➔ Kitchen)
```bash
cd src
python run_prototype_attention.py \
  --data_dir "../data/amazon/electronics-kitchen" \
  --output_dir "../run_out/amazon/electronics-kitchen" \
  --train_batch_size 10 --num_train_epochs 4 --learning_rate 2e-5 \
  --max_seq_length 256 --eval_batch_size 32 --labeled_num 1600 \
  --bert_model bert_base --seed 42 --do_train --do_eval
```

### 3. Launch the Streamlit Dashboard
Place your trained `attention_model.pt` checkpoint under `./run_out/amazon/<domain-pair>/attention_model.pt`, then run:
```bash
streamlit run app.py
```

---

## Author
**Naman Rathod** — M.Tech (Artificial Intelligence & Data Science), Netaji Subhas University of Technology (NSUT), New Delhi.
