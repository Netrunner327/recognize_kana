# Kana Recognition CNN

A lightweight TensorFlow/Keras CNN for recognizing handwritten hiragana and katakana characters, optimized for browser deployment via TensorFlow.js.

## Project Overview

This project trains a compact convolutional neural network to classify 92 Japanese kana characters (46 hiragana + 46 katakana). The model is designed to be converted to TensorFlow.js for client-side inference in a Next.js web application where users draw kana characters for practice.

## Project Structure

```
recognize_kana/
├── data/                      # Dataset and preprocessing
│   ├── raw/                   # Raw dataset files (ETL8G, ETL1)
│   ├── processed/             # Preprocessed .npz dataset
│   └── preprocess_data.py     # Extract, balance, augment, normalize pipeline
├── model/                     # Model architecture and training
│   ├── cnn_model.py          # CNN architecture definition
│   ├── train.py              # Training script
│   └── evaluate.py           # Model evaluation
├── scripts/                   # Utility scripts
│   ├── visualize_dataset.py  # Visualize training data
│   └── visualize_augmentation.py  # Check augmentation quality
├── converted_model/           # TensorFlow.js converted model
│   ├── model.json            # Model architecture
│   └── group1-shard*.bin     # Model weights
├── saved_models/              # Keras model checkpoints
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Download dataset (ETL Character Database)
# ETL8G for Hiragana and ETL1 for Katakana
http://etlcdb.db.aist.go.jp/the-etl-character-database/

# Extract ZIPs to data/raw/
unzip ETL8G*.zip -d data/raw/etl8g/
unzip ETL1*.zip -d data/raw/etl1/

# Preprocess: extract, balance, augment, normalize
python data/preprocess_data.py
```

### 3. Train Model

```bash
# Train CNN
python model/train.py --epochs 50 --batch-size 32 --input-size 64

# Evaluate model
python model/evaluate.py --model-path saved_models/best_model.h5
```

### 4. Convert to TensorFlow.js

```bash
# Convert trained model
tensorflowjs_converter \
    --input_format=keras \
    --quantize_uint8 \
    saved_models/best_model.h5 \
    converted_model/
```

## Model Architecture

- **Input:** 64x64 grayscale images
- **Architecture:** 
  - Conv2D(32, 3×3) → ReLU → MaxPool(2×2)
  - Conv2D(64, 3×3) → ReLU → MaxPool(2×2)
  - Conv2D(128, 3×3) → ReLU → MaxPool(2×2)
  - Flatten → Dense(256) → Dropout(0.5)
  - Dense(92) → Softmax
- **Parameters:** ~500K (optimized for web deployment)
- **Target Accuracy:** >95% on validation set


