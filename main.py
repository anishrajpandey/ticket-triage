import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report




print("Loading banking77 dataset...")
dataset = load_dataset("banking77")

train_data = dataset["train"]
test_data = dataset["test"]

# 2. Map the text and labels (no messy Pandas slicing required)
X_train = train_data["text"]
y_train = train_data["label"]

X_test = test_data["text"]
y_test = test_data["label"]

# 3. Apply your exact Vectorizer logic
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    stop_words='english',
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train your exact Logistic Regression model
print("Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)

model.fit(X_train_vec, y_train)

# 5. Evaluate and print the actual signal
print("Evaluating model...")
y_pred = model.predict(X_test_vec)

# Dynamically pull the 77 category names directly from the dataset metadata
target_names = train_data.features['label'].names

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"\nVocabulary Size: {len(vectorizer.vocabulary_)}")


user_input = "which card should i chose visa or mastercard?"  # Example user query about card choice

# 2. Vectorize it (IMPORTANT)
user_vec = vectorizer.transform([user_input])

# 3. Predict
pred_label = model.predict(user_vec)[0]
pred_proba = model.predict_proba(user_vec).max()

# 4. Convert label id to category name
pred_category = target_names[pred_label]

print(f"\nInput: {user_input}")
print(f"Predicted Category: {pred_category}")
print(f"Confidence Score: {pred_proba:.4f}")