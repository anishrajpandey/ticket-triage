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



# # 1. Create the inference pipeline
# # This connects the model and tokenizer into one "predictor" object
# classifier = pipeline(
#     "text-classification", 
#     model=model, 
#     tokenizer=tokenizer
# )

# # 2. Give it a raw string to categorize
# user_input = "I'm still waiting for my new card to arrive in the mail."

# # 3. Get the prediction
# prediction = classifier(user_input)

# print(f"\nInput: {user_input}")
# print(f"Predicted Category: {prediction[0]['label']}")
# print(f"Confidence Score: {prediction[0]['score']:.4f}")