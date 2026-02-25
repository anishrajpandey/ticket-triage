import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import json
import os
dataset = load_dataset("banking77")

X_train = dataset["train"]["text"]
y_train = dataset["train"]["label"]

label_names = dataset["train"].features["label"].names

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        stop_words='english',
        sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

LABEL_PATH = os.path.join(ROOT_DIR, "models", "label_names.json")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ticket_model.pkl")

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, MODEL_PATH)


with open(LABEL_PATH, "w") as f:
    json.dump(label_names, f)

