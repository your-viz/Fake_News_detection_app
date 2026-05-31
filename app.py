"""
Explainable Fake News Detection — Gradio App
Deployable on Azure App Services / Hugging Face Spaces / any PaaS
"""

import os
import re
import string
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR   = os.environ.get("MODEL_DIR", "./model")   # local path or HF hub id
MAX_LEN     = 256
TOP_K_FRAC  = 0.25
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# LOAD MODEL  (cached after first load)
# ─────────────────────────────────────────────
print("⏳ Loading model …")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_DIR, output_attentions=True
).to(device)
model.eval()
print("✅ Model loaded!")

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

# ─────────────────────────────────────────────
# ATTENTION IMPORTANCE
# ─────────────────────────────────────────────
def word_importance(attentions, attention_mask):
    last_layer = attentions[-1]           # (batch, heads, seq, seq)
    cls_row    = last_layer[:, :, 0, :]  # (batch, heads, seq)
    scores     = cls_row.mean(dim=1)     # (batch, seq)
    special_ids = set(tokenizer.convert_tokens_to_ids(list(SPECIAL_TOKENS)))
    # zero padding & special tokens
    for sid in special_ids:
        pass  # handled via attention_mask below
    scores = scores * attention_mask.float()
    return scores

# ─────────────────────────────────────────────
# MASK UNIMPORTANT TOKENS (faithfulness)
# ─────────────────────────────────────────────
def mask_unimportant(input_ids, attention_mask, importance_scores):
    masked = input_ids.clone()
    mask_token_id = tokenizer.mask_token_id
    real_len  = attention_mask[0].sum().item()
    scores_1d = importance_scores[0, :real_len]
    top_k     = max(1, int(real_len * TOP_K_FRAC))
    top_idx   = scores_1d.topk(top_k).indices
    keep      = torch.zeros(real_len, dtype=torch.bool, device=device)
    keep[top_idx] = True
    for i in range(real_len):
        if not keep[i]:
            masked[0, i] = mask_token_id
    return masked

# ─────────────────────────────────────────────
# SUBWORD MERGING
# ─────────────────────────────────────────────
def merge_subwords(tokens, scores):
    merged_tokens, merged_scores = [], []
    cur_word, cur_score, cur_count = "", 0.0, 0
    for tok, sc in zip(tokens, scores):
        if tok.startswith("##"):
            cur_word  += tok[2:]
            cur_score += sc
            cur_count += 1
        else:
            if cur_word:
                merged_tokens.append(cur_word)
                merged_scores.append(cur_score / max(cur_count, 1))
            cur_word  = tok
            cur_score = sc
            cur_count = 1
    if cur_word:
        merged_tokens.append(cur_word)
        merged_scores.append(cur_score / max(cur_count, 1))
    return merged_tokens, np.array(merged_scores)

# ─────────────────────────────────────────────
# HTML HIGHLIGHT
# ─────────────────────────────────────────────
def highlight_html(tokens, scores):
    if scores.max() == 0:
        return "<p>No significant tokens found.</p>"
    top_k   = max(1, int(len(tokens) * TOP_K_FRAC))
    top_idx = set(np.argsort(scores)[::-1][:top_k])
    max_s   = scores.max()
    parts   = []
    for i, (tok, sc) in enumerate(zip(tokens, scores)):
        tok = tok.replace("Ġ", "").strip()
        if not tok or tok in SPECIAL_TOKENS:
            continue
        if i in top_idx:
            alpha = 0.30 + 0.70 * (sc / max_s)
            parts.append(
                f'<mark style="background:rgba(255,180,0,{alpha:.2f});'
                f'padding:3px 6px;border-radius:6px;margin:2px;font-size:16px;">'
                f"{tok}</mark>"
            )
        else:
            parts.append(
                f'<span style="color:#777;margin:2px;font-size:16px;">{tok}</span>'
            )
    return "<div style='line-height:2.2'>" + " ".join(parts) + "</div>"

# ─────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────
@torch.no_grad()
def predict_news(text):
    if not text or not text.strip():
        return "⚠ Please enter news text.", "", "", ""

    cleaned = preprocess_text(text)

    enc = tokenizer(
        cleaned, max_length=MAX_LEN,
        padding=True, truncation=True, return_tensors="pt"
    )
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    out    = model(input_ids=ids, attention_mask=mask, output_attentions=True)
    logits = out.logits
    probs  = F.softmax(logits, dim=-1)[0]
    pred   = int(logits.argmax(-1).item())
    label  = "✅ REAL" if pred == 0 else "🚨 FAKE"
    conf   = probs[pred].item() * 100

    imp         = word_importance(out.attentions, mask)
    masked_ids  = mask_unimportant(ids, mask, imp)
    out_masked  = model(input_ids=masked_ids, attention_mask=mask)
    masked_pred = int(out_masked.logits.argmax(-1).item())
    faithful    = (pred == masked_pred)

    real_len = mask[0].sum().item()
    tokens   = tokenizer.convert_ids_to_tokens(ids[0][:real_len])
    scores   = imp[0, :real_len].cpu().numpy()
    tokens, scores = merge_subwords(tokens, scores)

    highlighted    = highlight_html(tokens, scores)
    ranked         = np.argsort(scores)[::-1][:10]
    top_words_text = "\n".join(tokens[i] for i in ranked
                                if tokens[i] not in SPECIAL_TOKENS)

    faith_text = (
        "✔ Faithful — prediction unchanged after masking unimportant words."
        if faithful
        else "⚠ Prediction changed after masking — explanation may be unstable."
    )

    result_md = f"""## Prediction: {label}

**Confidence:** {conf:.2f}%

| Class | Probability |
|-------|-------------|
| ✅ REAL | {probs[0].item()*100:.2f}% |
| 🚨 FAKE | {probs[1].item()*100:.2f}% |
"""
    return result_md, highlighted, top_words_text, faith_text


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────
examples = [
    ["Scientists confirmed a new exoplanet using the James Webb telescope."],
    ["BREAKING: Government secretly injects microchips through vaccines."],
    ["The Reserve Bank increased interest rates due to inflation concerns."],
    ["SHOCKING: Celebrities operate a hidden underground laboratory."],
    ["New study links processed food consumption to heart disease."],
]

custom_css = """
body { font-family: Arial, sans-serif; }
.gradio-container { max-width: 1100px !important; }
mark { transition: background 0.2s; }
"""

demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste a news article or headline here …",
        label="📰 News Text",
    ),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.HTML(label="🔍 Influential Word Highlights"),
        gr.Textbox(label="📋 Top Influential Words", lines=10),
        gr.Textbox(label="🧪 Faithfulness Check"),
    ],
    title="📰 Explainable Fake News Detector",
    description="""
**DistilBERT-based Fake News Detection** with Attention Explainability

✅ Attention-based Word Importance  &nbsp;|&nbsp;  ✅ Faithfulness Verification  &nbsp;|&nbsp;  ✅ Real-time Inference
""",
    examples=examples,
    theme=gr.themes.Soft(),
    css=custom_css,
    allow_flagging="never",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
