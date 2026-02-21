from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Load model once at startup
model = joblib.load("models/ticket_model.pkl")

with open("models/label_names.json") as f:
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
    top_idx = np.argmax(probs)

    confidence = float(probs[top_idx])
    category = label_names[top_idx]

    status = (
        "auto_routed"
        if confidence >= CONFIDENCE_THRESHOLD
        else "needs_human_review"
    )

    return {
        "predicted_category": category,
        "confidence": confidence,
        "status": status
        
    }




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)