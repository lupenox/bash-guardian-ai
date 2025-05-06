#!/bin/bash

echo "🐺 Activating virtual environment..."
source ../../venv/bin/activate

echo "📚 Installing dependencies..."
pip install transformers datasets accelerate

echo "🔥 Starting Bash GPT-2 fine-tuning..."
python3 train_gpt2.py
