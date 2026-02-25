from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os


app = FastAPI()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, "models", "ticket_model.pkl")
LABEL_PATH = os.path.join(ROOT_DIR, "models", "label_names.json")

# Load model once at startup
model = joblib.load(MODEL_PATH)


with open(LABEL_PATH) as f:
    label_names = json.load(f)

CONFIDENCE_THRESHOLD =0.3;


class TicketRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Ticket Triage API! Use the /predict endpoint to classify your tickets."}

@app.post("/predict")
def predict(ticket: TicketRequest):
    probs = model.predict_proba([ticket.text])[0]
    
    # Get top 5 indices
    top_indices = np.argsort(probs)[::-1][:5]
    
    top_idx = top_indices[0]
    confidence = float(probs[top_idx])
    category = label_names[top_idx]

    status = (
        "auto_routed"
        if confidence >= CONFIDENCE_THRESHOLD
        else "needs_human_review"
    )

    scores = {label_names[i]: float(probs[i]) for i in top_indices}

    return {
        "predicted_category": category,
        "confidence": confidence,
        "status": status,
        "scores": scores
    }




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)