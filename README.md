# Hybrid Stochastic-Discriminative Model for Credit Card Fraud Detection



## Overview
This repository implements a hybrid approach to credit card fraud detection that combines a **Hidden Markov Model (HMM)** for sequential (stochastic) feature generation with a **Random Forest (RF)** classifier to detect rare, non-linear anomalies in transactional data. The HMM models normal spending sequences and produces a log-likelihood anomaly score which is used as an additional feature for the discriminative RF classifier.

**Key highlights**
- Addresses extreme class imbalance (fraud < 0.2% of transactions).  
- Models sequential user behavior with a Gaussian HMM.  
- Uses Random Oversampling to balance training data and Random Forest for classification.  
- Achieves strong cross-validated performance (5-fold CV F1 ≈ 0.996, AUC ≈ 0.998).

---



## Project Structure

<pre>
.
├── creditcard_fraud_detection.ipynb     # Main Jupyter notebook (full pipeline)
├── data/                                # (not included) Place creditcard.csv here
├── requirements.txt                     # Python dependencies (optional)
└── README.md
</pre>





---

## Technology Stack
- **Python** (3.8+)  
- **pandas**, **NumPy** (data processing)  
- **scikit-learn** (Random Forest, metrics, CV)  
- **hmmlearn** (GaussianHMM for stochastic modeling)  
- **imbalanced-learn** (RandomOversampler)  
- **matplotlib**, **seaborn** (visualizations)

---

## Methodology

### Stage 1 — Stochastic Feature Engineering (HMM)
- Train a `GaussianHMM` on **normal (non-fraud)** transactions only to learn typical sequential behavior.  
- Compute the **Log-Likelihood Score** `HMM_LogLikelihood = log P(O | HMM)` for each transaction (or sequence window).  
- Lower (more negative) log-likelihood values indicate anomalous sequences and serve as a stochastic risk metric.

### Stage 2 — Discriminative Classification (Random Forest)
- Append `HMM_LogLikelihood` to the original feature set to form the hybrid feature matrix.  
- Handle class imbalance by **Random Oversampling** the minority class to a balanced training set.  
- Train a `RandomForestClassifier` on the hybrid features.  
- Evaluate using cross-validation and a held-out test set.

---

## Results (Representative)
**Cross-Validation (5-fold)**  
- Average F1-Score: **0.996**

**Test Set Metrics**
| Class      | Precision | Recall | F1-Score |
|------------|----------:|-------:|---------:|
| Non-Fraud (0) | 0.99     | 1.00   | 0.99    |
| Fraud (1)     | 1.00     | 0.99   | 0.99    |

**Feature Importance (Top 4)**
| Rank | Feature           | Importance |
|-----:|-------------------|----------:|
| 1    | V14               | 0.2099    |
| 2    | V10               | 0.1155    |
| 3    | HMM_LogLikelihood | 0.1011    |
| 4    | V4                | 0.0972    |

> The HMM log-likelihood ranks highly, confirming that stochastic sequential features provide unique predictive power.

---

## Setup & Usage

1. Clone this repository:
```bash
git clone https://github.com/<your-username>/Hybrid-Stochastic-ML.git
cd Hybrid-Stochastic-ML

2. Install dependencies (recommended: virtualenv / conda):
pip install -r requirements.txt
# or
pip install pandas numpy scikit-learn imbalanced-learn hmmlearn matplotlib seaborn

3. Download the Kaggle Credit Card Fraud dataset (creditcard.csv) and place it in ./data/.

4. Open and run the main notebook:

creditcard_fraud_detection.ipynb — runs the full pipeline (data load → HMM feature generation → oversampling → RF training → evaluation).
