# Customer Support Ticket Triage API

A deployed machine learning microservice that classifies incoming customer support tickets into predefined departments using classical NLP techniques.

**Live Demo:**  
<a href="https://anishrajpandey.github.io/ticket-triage/" target="_blank">https://anishrajpandey.github.io/ticket-triage </a>

https://anishrajpandey.github.io/ticket-triage/

<img width="1148" height="673" alt="image" src="https://github.com/user-attachments/assets/7ebeca0c-0ecf-4d9a-a87f-e5dac48df2cc" />

**Live API:**  

<a href="https://ticket-triage-lltp.onrender.com/" target="_blank"> https://ticket-triage-lltp.onrender.com </a>




---

## Overview

Customer support systems receive large volumes of unstructured text requests that must be routed to the correct department (e.g., Billing, Technical Support, Account Issues). Manual triaging does not scale and introduces routing errors.

This project implements a production-style text classification service that:

- Predicts ticket category  
- Returns prediction confidence  
- Routes low-confidence cases to human review  

The objective is not just high accuracy, but reducing costly misrouting in real workflows.

---

## System Architecture
Client (UI / curl)
↓
FastAPI REST Service
↓
TF-IDF Vectorizer
↓
Logistic Regression (Multiclass)
↓
Confidence Threshold Logic



### Components

- **Backend:** FastAPI + Gunicorn (deployed on Render)
- **Model:** Scikit-learn pipeline (TF-IDF + Logistic Regression)
- **Frontend:** Static demo UI (GitHub Pages)
- **Training Script:** Offline training module
- **Artifacts:** Serialized model + label map stored in `/models`

The API loads trained artifacts at startup and serves inference requests.

---

## API Usage

### Endpoint

`POST /predict`

### Example Request

```json
{
  "text": "My internet connection keeps dropping"
}
```
Example Response (Auto Assigned)
```json
{
  "predicted_category": "Technical Support",
  "confidence": 0.87,
  "status": "auto_assigned"
}
```

Example Response (Human Review)
```json
{
  "predicted_category": "Billing",
  "confidence": 0.42,
  "status": "human_review"
}
```
### Text Representation

TF-IDF vectorization
Unigrams + bigrams
Sparse feature matrix

**Model**: Multinomial Logistic Regression

Why not transformers?

- [ ] The problem scope does not require deep contextual modeling.

- [ ] Classical ML is faster to train, cheaper to deploy, and easier to debug.

- [ ] The objective is reliable routing, not semantic generation.


**Deployment**
- Backend Hosted on Render (Python Web Service)
- Static UI Hosted with Github Pages
- Interactive API docs available at /docs

## Limitations

- No automatic response generation
- No multilingual support
- No ticket prioritization
- No retraining pipeline
- No database persistence


## Local Development
```python

pip install -r requirements.txt

uvicorn app.main:app --reload
```

Visit : 
```code
GET http://127.0.0.1:8000/docs
POST http://127.0.0.1:8000/predict

```





##Summary

- This project demonstrates:
- End-to-end ML workflow (training → serialization → deployment)
- Clean separation between training and inference
- Confidence-aware routing logic
- Publicly deployed ML microservice
- Production-style dependency management

Developed with ❤️ by Anish. Feel free to fork, star and give me feedback on my project. 

