#!/bin/bash
# startup.sh — Azure App Service startup command
# Set in Azure Portal → Configuration → Startup Command:
#   bash startup.sh

pip install -r requirements.txt --quiet
python app.py
