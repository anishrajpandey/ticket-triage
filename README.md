# Customer Support Ticket Triage

A production-oriented NLP project that classifies incoming customer support tickets into the correct department using classical machine learning techniques. The system includes confidence-based routing to defer low-confidence predictions to human review.

---

## Problem Statement

Customer support teams receive large volumes of unstructured text tickets that must be routed to the correct department (e.g. Billing, Technical Support, Account Issues). Manual triaging is slow, error-prone, and does not scale.

This project builds a **multiclass text classification system** to automatically route tickets while minimizing the risk of incorrect assignments.

---

## Scope (V1)

- Pure text classification (no LLMs, no transformers)
- Multiclass routing into predefined support categories
- Confidence-based fallback to human review
- Model served via a REST API

**Out of scope**
- Auto-generated responses
- Ticket prioritization
- Language translation
- Large language models

---

## Dataset

The model is trained on a public customer support ticket dataset containing:
- Free-form text (subject + description)
- Pre-labeled support categories

The dataset is intentionally noisy to reflect real-world customer input.

---

## Approach

### Text Representation
- TF-IDF vectorization
- Unigrams and bigrams

### Models
- Logistic Regression (multinomial) as baseline
- Linear classifiers chosen for interpretability and efficiency

### Evaluation Metrics
- Macro F1-score (primary)
- Per-class precision and recall
- Confusion matrix

Accuracy is not used as a primary metric due to class imbalance.

---

## Confidence-Based Routing

Predictions below a configurable confidence threshold are routed to **human review** instead of being auto-assigned.

This reduces misrouting risk and reflects real-world deployment constraints.

---

API : https://ticket-triage-lltp.onrender.com/
UI : tbd
