# Hybrid Sequential ML Model for Credit Card Fraud Detection

A hybrid machine learning pipeline for detecting fraudulent credit card transactions using **Gaussian Hidden Markov Models (HMMs)**, **Random Forest**, and **XGBoost** on highly imbalanced financial transaction data.

This project combines:
- Sequential behavioral modeling using HMMs
- Ensemble machine learning classifiers
- Advanced feature engineering
- Threshold optimization
- Business-impact evaluation for fraud-risk tradeoffs

---

# Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions constitute only a tiny fraction of total transactions.

Traditional ML models often struggle to:
- capture sequential behavioral patterns,
- handle extreme class imbalance,
- optimize business tradeoffs between fraud loss and false declines.

This project addresses these issues by integrating:
1. **Gaussian Hidden Markov Models (HMMs)** for transaction sequence modeling
2. **Tree-based ensemble classifiers** for high-performance tabular fraud classification

---

# Key Features

## Sequential Behavioral Modeling
- Trained Gaussian Hidden Markov Models on cardholder transaction sequences
- Modeled hidden behavioral states of users
- Generated HMM likelihood scores as additional fraud-risk features

## Ensemble ML Models
Implemented and compared:
- Random Forest
- XGBoost
- HMM-Augmented Hybrid Models

## Feature Engineering
Engineered:
- Temporal transaction features
- Behavioral spending features
- Geospatial distance features
- Cardholder activity patterns

## Imbalance Handling
- Fraud rate ≈ 0.579%
- Class imbalance ratio ≈ 172:1
- Stratified splitting and threshold optimization applied

## Evaluation Pipeline
Evaluated models using:
- ROC-AUC
- PR-AUC
- F1 Score
- Precision / Recall
- Confusion Matrices
- Business-impact analysis

---

# Dataset

## Source
Kaggle Credit Card Fraud Detection Dataset

## Dataset Statistics
- ~1.29 million transaction records
- 23 transaction and behavioral features
- Fraud rate: ~0.579%
- Highly imbalanced binary classification problem

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- hmmlearn
- Matplotlib
- Seaborn

---

# Project Workflow

```text
Raw Transaction Data
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Gaussian HMM Training
        ↓
Generate HMM Sequence Scores
        ↓
Random Forest / XGBoost Training
        ↓
Threshold Optimization
        ↓
Evaluation & Business Impact Analysis
```

---

# Hidden Markov Model (HMM) Integration

The project uses Gaussian HMMs to model sequential transaction behavior.

For each cardholder:
- transactions are treated as sequences,
- hidden behavioral states are learned,
- sequence likelihood scores are generated,
- scores are appended as additional ML features.

This enables the model to capture:
- abnormal transaction behavior,
- deviations from normal spending patterns,
- temporal dependencies.

---

# Models Implemented

| Model | Description |
|---|---|
| Random Forest Baseline | Standard RF classifier |
| Random Forest + HMM | RF augmented with HMM sequence scores |
| XGBoost Baseline | Gradient boosting fraud classifier |
| XGBoost + HMM | Hybrid sequential fraud detection model |

---

# Results

| Model | ROC-AUC |
|---|---|
| Random Forest Baseline | 0.9921 |
| Random Forest + HMM | 0.9912 |
| XGBoost Baseline | 0.9986 |
| XGBoost + HMM | 0.9986 |

## Best F1 Score
- F1 Score ≈ 0.876 after threshold optimization

---

# Business Impact Evaluation

Beyond traditional ML metrics, the project evaluates:
- fraud-loss reduction,
- false-decline costs,
- operational tradeoffs.

This simulates real-world financial fraud detection systems where:
- overly aggressive fraud blocking hurts customer experience,
- overly relaxed detection increases fraud losses.

---

# Visualizations Included

- Fraud distribution analysis
- Temporal fraud trends
- Correlation heatmaps
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Feature importance plots

---



---

# Installation

Clone the repository:

```bash
git clone [https://github.com/yourusername/hybrid-fraud-detection-ml.git](https://github.com/Pranathi3101/Hybrid-Stochastic-ML-Model-for-Credit-Card-Fraud-Detection.git)
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Open the notebook:

```bash
jupyter notebook
```

Run:

```text
notebook.ipynb
```

---

# Future Improvements

- Graph Neural Networks (GNNs)
- Real-time streaming fraud detection
- Transformer-based sequence modeling
- Autoencoder anomaly detection
- Online learning for evolving fraud patterns

---



