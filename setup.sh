#!/bin/bash
set -e
echo "[bio-compute-stack] Setting up environment..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete."
