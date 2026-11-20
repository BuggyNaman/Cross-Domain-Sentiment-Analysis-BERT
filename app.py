import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _here]
import os
import html
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertTokenizer

# ==============================================================================
# 1. Model Architecture (Exact match to run_prototype_attention.py)
# ==============================================================================

class TokenAttention(nn.Module):
    """Learns token-level importance weights over BERT's sequence output."""
    def __init__(self, hidden_size):
        super(TokenAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, sequence_output, attention_mask):
        scores = self.attn(sequence_output).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), sequence_output).squeeze(1)
        return context, weights


class IntermediateMLP(nn.Module):
    def __init__(self, config):
        super(IntermediateMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU()
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)


class BertForDomain(BertPreTrainedModel):
    all_tied_weights_keys = {}
    _tied_weights_keys = []

    def __init__(self, config):
        super(BertForDomain, self).__init__(config)
        config.return_dict = False
        self.bert = BertModel(config)
        self.domain_attn = TokenAttention(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.domain_classifier = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, return_weights=False):
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        sequence_output = outputs[0]
        pooled_output, domain_weights = self.domain_attn(sequence_output, attention_mask)
        pooled_output = self.dropout(pooled_output)
        if return_weights:
            return pooled_output, domain_weights
        return pooled_output


class BertaForClassificationModel(BertPreTrainedModel):
    all_tied_weights_keys = {}
    _tied_weights_keys = []

    def __init__(self, config, num_labels=2, *model_args, **model_kwargs):
        super(BertaForClassificationModel, self).__init__(config)
        config.return_dict = False
        self.bert = BertModel(config)
        self.config = config
        self.num_labels = num_labels

        self.sent_attn = TokenAttention(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_feature = IntermediateMLP(config)
        self.sent_classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.domain = BertForDomain(config)

        self.KL_loss = nn.KLDivLoss(reduction="batchmean")
        self.CE_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src_input_ids=None, src_attention_mask=None, return_weights=True):
        outputs = self.bert(src_input_ids, attention_mask=src_attention_mask, return_dict=False)
        src_pooled, sent_weights = self.sent_attn(outputs[0], src_attention_mask)
        src_sent_feature = self.sentiment_feature(self.dropout(src_pooled))
        src_domain_feature, domain_weights = self.domain(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
            return_weights=True
        )
        logits = self.sent_classifier(src_sent_feature)
        if return_weights:
            return logits, sent_weights, domain_weights
        return src_pooled, src_sent_feature, src_domain_feature, logits


# ==============================================================================
# 2. Caching & Inference Helpers
# ==============================================================================

@st.cache_resource(show_spinner="Loading BERT tokenizer...")
def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


@st.cache_resource(show_spinner="Loading trained Disentangled Attention checkpoint...")
def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.return_dict = False
    model = BertaForClassificationModel(config, num_labels=2)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device


def merge_wordpieces(tokens, sent_w, dom_w):
    """Merge WordPiece subwords (##) into full words and drop special tokens."""
    words, s_weights, d_weights = [], [], []
    for tok, sw, dw in zip(tokens, sent_w, dom_w):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith("##") and words:
            words[-1] += tok[2:]
            s_weights[-1] += float(sw)
            d_weights[-1] += float(dw)
        else:
            words.append(tok)
            s_weights.append(float(sw))
            d_weights.append(float(dw))
    return words, np.array(s_weights, dtype=float), np.array(d_weights, dtype=float)


def render_heatmap_html(words, weights, rgb_tuple):
    """Render word-level attention heatmap as clean styled HTML spans."""
    if len(weights) == 0:
        return ""
    w_min, w_max = float(np.min(weights)), float(np.max(weights))
    denom = (w_max - w_min) if (w_max - w_min) > 1e-8 else 1.0
    norm_w = (weights - w_min) / denom

    r, g, b = rgb_tuple
    spans = []
    for word, score in zip(words, norm_w):
        alpha = 0.08 + 0.77 * float(score)
        text_color = "#ffffff" if alpha > 0.55 else "#111827"
        safe_word = html.escape(word)
        spans.append(
            f'<span title="Attention score: {score:.3f}" style="'
            f'background-color: rgba({r}, {g}, {b}, {alpha:.2f}); '
            f'color: {text_color}; padding: 3px 6px; margin: 3px 2px; '
            f'border-radius: 5px; display: inline-block; font-size: 0.98rem; '
            f'line-height: 1.5;">{safe_word}</span>'
        )
    return (
        '<div style="background-color: #f9fafb; border: 1px solid #e5e7eb; '
        'border-radius: 10px; padding: 14px; line-height: 2.1;">'
        + " ".join(spans)
        + "</div>"
    )


def predict_review(text, model, tokenizer, device, max_len=256):
    encoded = tokenizer(
        text.lower(),
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits, sent_w, dom_w = model(
            src_input_ids=input_ids,
            src_attention_mask=attention_mask,
            return_weights=True
        )
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    seq_len = int(attention_mask.sum().item())
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0)[:seq_len].cpu().tolist())
    sw = sent_w.squeeze(0)[:seq_len].cpu().numpy()
    dw = dom_w.squeeze(0)[:seq_len].cpu().numpy()

    words, word_sw, word_dw = merge_wordpieces(tokens, sw, dw)
    pred_idx = int(np.argmax(probs))
    sentiment = "POSITIVE" if pred_idx == 1 else "NEGATIVE"
    confidence = float(probs[pred_idx])
    return sentiment, confidence, probs, words, word_sw, word_dw


# ==============================================================================
# 3. Page Configuration & Static Benchmark Data
# ==============================================================================

st.set_page_config(
    page_title="Cross-Domain Sentiment Analysis | Disentangled Attention",
    page_icon="🔍",
    layout="wide"
)

DOMAIN_PAIRS = {
    "Electronics ➔ Kitchen (Best: 93.75%)": "electronics-kitchen",
    "Books ➔ DVDs (89.40%)": "books-dvs",
    "DVDs ➔ Electronics (89.20%)": "dvs-electronics",
    "Kitchen ➔ Books (84.25%)": "kitchen-books",
}

PRESET_REVIEWS = {
    "Positive Kitchen Review": (
        "This stainless steel blender is absolutely fantastic. The motor runs quietly, "
        "the blades crush ice effortlessly, and cleanup takes less than a minute. "
        "Highly durable and worth every penny."
    ),
    "Negative Electronics Review": (
        "Terrible wireless router. The signal drops every ten minutes, the setup "
        "software kept crashing, and the battery pack overheated on the second day. "
        "Complete waste of money."
    ),
    "Positive Book Review": (
        "An unforgettable novel with brilliant pacing and deeply compelling characters. "
        "The author weaves suspense into every chapter and delivers a genuinely "
        "satisfying ending."
    ),
    "Mixed / Nuanced Review": (
        "The cinematography and soundtrack on this DVD were visually stunning, "
        "but the plot felt painfully slow and the dialogue was awkward and boring "
        "throughout the second half."
    ),
}

BENCHMARK_DF = pd.DataFrame([
    {"Transfer Task": "Books ➔ DVDs", "BERT Baseline (%)": 86.10, "Proposed Attention Model (%)": 89.40, "Improvement (Δ)": "+3.30%"},
    {"Transfer Task": "DVDs ➔ Electronics", "BERT Baseline (%)": 87.85, "Proposed Attention Model (%)": 89.20, "Improvement (Δ)": "+1.35%"},
    {"Transfer Task": "Electronics ➔ Kitchen", "BERT Baseline (%)": 92.25, "Proposed Attention Model (%)": 93.75, "Improvement (Δ)": "+1.50%"},
    {"Transfer Task": "Kitchen ➔ Books", "BERT Baseline (%)": 81.05, "Proposed Attention Model (%)": 84.25, "Improvement (Δ)": "+3.20%"},
    {"Transfer Task": "Average", "BERT Baseline (%)": 86.81, "Proposed Attention Model (%)": 89.15, "Improvement (Δ)": "+2.34%"},
])

DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/NSUT/sem1/Aiml/Project/CDSA_Replication/disentangled_representation/run_out/amazon"
DEFAULT_LOCAL_DIR = "./run_out/amazon"
DEFAULT_BASE_DIR = DEFAULT_DRIVE_DIR if os.path.exists(DEFAULT_DRIVE_DIR) else DEFAULT_LOCAL_DIR


# ==============================================================================
# 4. Sidebar Controls
# ==============================================================================

with st.sidebar:
    st.title("⚙️ Controls")
    selected_label = st.selectbox("Select Cross-Domain Model", list(DOMAIN_PAIRS.keys()))
    selected_pair = DOMAIN_PAIRS[selected_label]

    base_ckpt_dir = st.text_input("Checkpoints Root Folder", value=DEFAULT_BASE_DIR)
    ckpt_path = os.path.join(base_ckpt_dir, selected_pair, "attention_model.pt")

    if os.path.exists(ckpt_path):
        st.success(f"Checkpoint found: `{selected_pair}/attention_model.pt`")
    else:
        st.warning(f"Checkpoint not found at `{ckpt_path}`")

    st.divider()
    st.subheader("⚡ One-Click Demo Reviews")
    if "review_input" not in st.session_state:
        st.session_state["review_input"] = PRESET_REVIEWS["Positive Kitchen Review"]

    for preset_name, preset_text in PRESET_REVIEWS.items():
        if st.button(preset_name, use_container_width=True):
            st.session_state["review_input"] = preset_text

    st.divider()
    st.subheader("🧠 Architecture Specs")
    st.markdown(
        "- **Backbone:** `bert-base-uncased`\n"
        "- **Attention:** Dual Token-Level Attention (Sentiment vs. Domain)\n"
        "- **Disentanglement:** KL Divergence + Active Orthogonal Loss\n"
        "- **Target Alignment:** Confidence-Gated Prototypical Loss ($\\tau \\ge 0.85$, Epoch $\\ge 2$)"
    )


# ==============================================================================
# 5. Main Header & Two-Tab Layout
# ==============================================================================

st.title("🔍 Cross-Domain Sentiment Analysis")
st.caption("Disentangled Token-Level Attention, Active Orthogonal Regularization & Confidence-Gated Prototypical Learning")

tab1, tab2 = st.tabs(["🧪 Live Cross-Domain Analyzer", "📊 Benchmarks & Architecture"])

# ------------------------------------------------------------------------------
# TAB 1: Live Analyzer
# ------------------------------------------------------------------------------
with tab1:
    review_text = st.text_area(
        "Enter or paste any product review:",
        key="review_input",
        height=130,
        placeholder="Type a review from Books, DVDs, Electronics, or Kitchen..."
    )

    analyze_clicked = st.button("🚀 Analyze Sentiment & Disentangle Attention", type="primary")

    if analyze_clicked:
        if not review_text.strip():
            st.error("Please enter a review before analyzing.")
        elif not os.path.exists(ckpt_path):
            st.error(
                f"Cannot find model weights at `{ckpt_path}`. "
                "Update the **Checkpoints Root Folder** in the sidebar to point to your `run_out/amazon` directory."
            )
        else:
            tokenizer = load_tokenizer()
            model, device = load_model(ckpt_path)
            sentiment, confidence, probs, words, word_sw, word_dw = predict_review(
                review_text, model, tokenizer, device
            )

            # Prediction Summary Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                badge = "🟢 POSITIVE" if sentiment == "POSITIVE" else "🔴 NEGATIVE"
                st.metric("Predicted Sentiment", badge)
            with col2:
                st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                st.progress(min(max(confidence, 0.0), 1.0))
            with col3:
                st.metric("Active Transfer Model", selected_pair.replace("-", " ➔ ").title())

            st.divider()

            # Disentangled Word-Level Heatmaps
            st.subheader("🎨 Disentangled Word-Level Attention Heatmaps")
            st.caption("Hover over any word to inspect its normalized attention weight.")

            sent_rgb = (22, 163, 74) if sentiment == "POSITIVE" else (220, 38, 38)
            dom_rgb = (79, 70, 229)

            st.markdown("**1. Sentiment Attention Heatmap** *(Domain-invariant opinion cues)*")
            st.markdown(render_heatmap_html(words, word_sw, sent_rgb), unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**2. Domain Attention Heatmap** *(Product-specific context isolated from sentiment)*")
            st.markdown(render_heatmap_html(words, word_dw, dom_rgb), unsafe_allow_html=True)

            st.divider()

            # Top Weighted Words Breakdown
            st.subheader("📈 Top-5 Weighted Tokens Breakdown")
            stop_punct = {".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")"}
            filtered = [
                (w, s, d) for w, s, d in zip(words, word_sw, word_dw)
                if w not in stop_punct and len(w) > 1
            ]

            if filtered:
                top_sent = sorted(filtered, key=lambda x: x[1], reverse=True)[:5]
                top_dom = sorted(filtered, key=lambda x: x[2], reverse=True)[:5]

                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown("**Top Sentiment-Driving Words**")
                    df_s = pd.DataFrame(
                        {"Word": [x[0] for x in top_sent], "Sentiment Weight": [x[1] for x in top_sent]}
                    ).set_index("Word")
                    st.bar_chart(df_s)
                with c_right:
                    st.markdown("**Top Domain-Specific Words**")
                    df_d = pd.DataFrame(
                        {"Word": [x[0] for x in top_dom], "Domain Weight": [x[2] for x in top_dom]}
                    ).set_index("Word")
                    st.bar_chart(df_d)

# ------------------------------------------------------------------------------
# TAB 2: Benchmarks & Architecture
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🏆 4-Pair Cross-Domain Amazon Benchmark Results")
    st.dataframe(BENCHMARK_DF, use_container_width=True, hide_index=True)

    st.subheader("📊 Baseline vs. Proposed Attention Architecture")
    chart_df = BENCHMARK_DF[BENCHMARK_DF["Transfer Task"] != "Average"].set_index("Transfer Task")[
        ["BERT Baseline (%)", "Proposed Attention Model (%)"]
    ]
    st.bar_chart(chart_df)

    st.divider()
    st.subheader("🛠️ Key Architectural Contributions")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**1. Dual Token-Level Attention**\n\n"
            "Replaces static `[CLS]` pooling with separate learned attention heads over all BERT tokens, "
            "explicitly focusing on sentiment adjectives while isolating domain nouns."
        )
    with c2:
        st.info(
            "**2. Active Orthogonal Loss**\n\n"
            "Penalizes squared cosine similarity between sentiment and domain feature vectors alongside "
            "KL divergence, mathematically enforcing feature disentanglement."
        )
    with c3:
        st.info(
            "**3. Confidence-Gated Prototypes**\n\n"
            "Anchors target alignment to high-confidence pseudo-labels ($\\tau \\ge 0.85$) starting at Epoch 2, "
            "preventing mode collapse on distant shifts like `DVDs ➔ Electronics`."
        )
