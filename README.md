# 📰 Explainable Fake News Detector

A DistilBERT-based fake news detection web app with attention explainability,
deployable on **Microsoft Azure App Service**.

## Features

- 🤖 Fine-tuned DistilBERT for binary classification (REAL / FAKE)
- 🔍 Attention-based word importance highlighting
- 🧪 Faithfulness verification (masked-prediction consistency)
- ⚡ Gradio web UI — no frontend code required

---

## Project Structure

```
fakenews_app/
├── app.py               ← Gradio app (entry point)
├── requirements.txt     ← Python dependencies
├── startup.sh           ← Azure startup command
├── .gitattributes       ← Git LFS config for model weights
├── .gitignore
└── model/               ← Place trained model files here
    └── README.md        ← Instructions for model weights
```

---

## Quick Local Test

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:7860
```

---

## Deploying to Azure App Service

See the full step-by-step guide in [DEPLOY_AZURE.md](DEPLOY_AZURE.md).

**TL;DR:**
1. Upload model weights to the `model/` folder (use Git LFS for large files).
2. Push to GitHub.
3. Create an Azure App Service (Python 3.11, B2 tier or higher).
4. Connect GitHub repo for CI/CD.
5. Set Startup Command to `bash startup.sh`.
6. Set `PORT=8000` in Application Settings.

---

## Environment Variables

| Variable    | Default      | Description                                    |
|-------------|--------------|------------------------------------------------|
| `MODEL_DIR` | `./model`    | Path or HuggingFace Hub ID for model weights   |
| `PORT`      | `7860`       | Port the app listens on                        |

---

## Model Weights

The trained model is **not** stored in this repo due to GitHub's 100 MB file limit.

**Option A — Git LFS** (weights inside this repo):
```bash
git lfs install
git lfs track "model/*.safetensors"
# then add & commit normally
```

**Option B — HuggingFace Hub** (recommended for Azure):
1. Push weights to HuggingFace: `model.push_to_hub("your-username/fakenews-distilbert")`
2. Set `MODEL_DIR=your-username/fakenews-distilbert` in Azure App Settings.

---

## License

MIT
