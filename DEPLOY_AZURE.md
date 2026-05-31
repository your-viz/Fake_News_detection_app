# 🚀 Deploy Fake News Detector on Azure App Service — Step-by-Step

---

## Prerequisites

- Microsoft Azure account ([free tier](https://azure.microsoft.com/free/) works)
- GitHub account
- Git + Git LFS installed locally
- Python 3.11 installed locally (for local testing)

---

## Step 1 — Prepare Your Model Files

Your training notebook saves the model with:
```python
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

This produces these files inside the saved directory:
```
config.json
tokenizer_config.json
tokenizer.json
vocab.txt
special_tokens_map.json
model.safetensors   (or pytorch_model.bin — ~260 MB)
```

Extract your downloaded zip (`ImprovedExfakenews_model-*.zip`) and
copy the contents into the `model/` folder of this project.

---

## Step 2 — Push to GitHub with Git LFS

Model weight files are >100 MB so GitHub blocks them without LFS.

```bash
# 1. Install Git LFS (one-time)
git lfs install

# 2. Clone or init your repo
git init
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 3. LFS tracking is already set in .gitattributes
#    Verify:
cat .gitattributes
# Should show: model/*.safetensors filter=lfs ...

# 4. Add everything
git add .
git commit -m "Initial commit — fake news detector app"

# 5. Push (LFS uploads weights automatically)
git push -u origin main
```

> **Alternative (recommended for large models):**
> Upload weights to HuggingFace Hub instead (see Step 2b below).
> Then you don't need LFS at all.

### Step 2b — HuggingFace Hub (optional but recommended)

```python
# Run this in your Colab notebook after training
model.push_to_hub("YOUR_HF_USERNAME/fakenews-distilbert")
tokenizer.push_to_hub("YOUR_HF_USERNAME/fakenews-distilbert")
```

Then skip copying weights into `model/` — you will set `MODEL_DIR`
as an Azure environment variable in Step 5.

---

## Step 3 — Create an Azure App Service

1. Go to [portal.azure.com](https://portal.azure.com)
2. Click **Create a resource** → search **Web App** → click **Create**
3. Fill in the form:

| Field | Value |
|-------|-------|
| Subscription | Your subscription |
| Resource Group | Create new → e.g. `fakenews-rg` |
| Name | `fakenews-detector` *(must be globally unique)* |
| Publish | **Code** |
| Runtime stack | **Python 3.11** |
| Operating System | **Linux** |
| Region | e.g. `East US` |
| Linux Plan | Create new |
| Pricing tier | **B2** or higher *(B1 often OOM for DistilBERT)* |

4. Click **Review + create** → **Create**
5. Wait ~2 minutes for deployment to complete.

---

## Step 4 — Connect GitHub for Continuous Deployment

1. In your new App Service, go to **Deployment Center** (left sidebar)
2. Source: **GitHub**
3. Click **Authorize** and sign in to GitHub
4. Choose:
   - Organization: your GitHub username
   - Repository: your repo name
   - Branch: `main`
5. Build provider: **GitHub Actions** (recommended) or **App Service Build Service**
6. Click **Save**

Azure will now auto-deploy every time you push to `main`.

---

## Step 5 — Configure Application Settings

1. In the App Service, go to **Configuration** → **Application Settings**
2. Click **+ New application setting** for each:

| Name | Value |
|------|-------|
| `PORT` | `8000` |
| `MODEL_DIR` | `./model` *(or your HuggingFace repo ID if using HF Hub)* |

3. Click **Save** (the app restarts automatically)

---

## Step 6 — Set the Startup Command

1. In the App Service, go to **Configuration** → **General Settings**
2. Find **Startup Command** and enter:
   ```
   bash startup.sh
   ```
3. Click **Save**

---

## Step 7 — Configure Port Forwarding

Azure App Service routes external HTTP traffic to the port your app
listens on. Gradio must bind to `0.0.0.0` (not `127.0.0.1`).
The `app.py` already does this:
```python
demo.launch(server_name="0.0.0.0", server_port=port)
```

Azure automatically handles HTTPS termination — your public URL will be
`https://fakenews-detector.azurewebsites.net`.

---

## Step 8 — Monitor & Test

1. Go to **Overview** in the App Service
2. Wait for the **Status** to show **Running**
3. Click the URL: `https://fakenews-detector.azurewebsites.net`
4. You should see the Gradio UI load in 30–60 seconds (cold start)

To view logs:
- **Log stream**: App Service → **Log stream** (left sidebar)
- **Kudu console**: `https://fakenews-detector.scm.azurewebsites.net`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| App crashes with OOM | Upgrade to B3 tier (4 GB RAM) |
| `Model directory not found` | Check `MODEL_DIR` env var points to correct path |
| 502 Bad Gateway | App is still starting; wait 60–90 s and refresh |
| `pip install` fails | Check `requirements.txt` has correct package versions |
| Large file rejected by GitHub | Enable Git LFS (`git lfs install`) |

---

## Cost Estimate

| Tier | RAM | vCPU | ~Monthly Cost |
|------|-----|------|--------------|
| B1   | 1.75 GB | 1 | ~$13 USD |
| B2   | 3.5 GB  | 2 | ~$27 USD ← recommended |
| B3   | 7 GB    | 4 | ~$54 USD |

Free tier (F1) has only 1 GB RAM — **not enough** for DistilBERT.

---

## Sharing Your App

Once deployed, share the URL:
```
https://fakenews-detector.azurewebsites.net
```

Anyone with the link can use the app — no login required.

To add a custom domain, go to App Service → **Custom domains**.
