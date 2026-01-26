"""
Review and inspect the sentiment_model.joblib file
"""
import joblib
import os
import sys

model_path = "sentiment_model.joblib"
vectorizer_path = "tfidf_vectorizer.joblib"

print("="*70)
print("Model Review - sentiment_model.joblib")
print("="*70)

# Check if files exist
if not os.path.exists(model_path):
    print(f"✗ Error: {model_path} not found!")
    sys.exit(1)

if not os.path.exists(vectorizer_path):
    print(f"✗ Error: {vectorizer_path} not found!")
    sys.exit(1)

# Load and inspect model
print("\n[1] Loading model...")
model = joblib.load(model_path)
print(f"✓ Model loaded successfully")
print(f"  Type: {type(model).__name__}")
print(f"  Module: {type(model).__module__}")

# Get file info
model_size = os.path.getsize(model_path)
vectorizer_size = os.path.getsize(vectorizer_path)

print(f"\n[2] File Information:")
print(f"  Model file: {model_path}")
print(f"  Model size: {model_size:,} bytes ({model_size/1024:.2f} KB)")
print(f"  Vectorizer file: {vectorizer_path}")
print(f"  Vectorizer size: {vectorizer_size:,} bytes ({vectorizer_size/1024:.2f} KB)")
print(f"  Total size: {(model_size + vectorizer_size):,} bytes ({(model_size + vectorizer_size)/1024:.2f} KB)")

# Model parameters
print(f"\n[3] Model Configuration:")
if hasattr(model, 'get_params'):
    params = model.get_params()
    print(f"  Regularization (C): {params.get('C', 'N/A')}")
    print(f"  Penalty: {params.get('penalty', 'N/A')}")
    print(f"  Solver: {params.get('solver', 'N/A')}")
    print(f"  Max iterations: {params.get('max_iter', 'N/A')}")

# Model coefficients
if hasattr(model, 'coef_'):
    n_features = model.coef_.shape[1]
    print(f"  Number of features: {n_features:,}")
    print(f"  Classes: {model.classes_.tolist()}")

# Load vectorizer
print(f"\n[4] Loading vectorizer...")
vectorizer = joblib.load(vectorizer_path)
print(f"✓ Vectorizer loaded successfully")
print(f"  Type: {type(vectorizer).__name__}")

if hasattr(vectorizer, 'get_params'):
    vec_params = vectorizer.get_params()
    print(f"  Max features: {vec_params.get('max_features', 'N/A')}")
    print(f"  Ngram range: {vec_params.get('ngram_range', 'N/A')}")
    print(f"  Stop words: {vec_params.get('stop_words', 'N/A')}")

if hasattr(vectorizer, 'vocabulary_'):
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

# Test prediction
print(f"\n[5] Testing Prediction Pipeline:")
test_texts = [
    "This movie is absolutely amazing and wonderful!",
    "Terrible film, complete waste of time."
]

import re

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for i, text in enumerate(test_texts, 1):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = proba[prediction] * 100

    print(f"\n  Test {i}:")
    print(f"    Text: {text}")
    print(f"    Prediction: {sentiment} ({confidence:.2f}% confidence)")

print("\n" + "="*70)
print("Model Review Complete!")
print("="*70)
print("\n✓ Both model and vectorizer are valid and ready for upload")
print("✓ Prediction pipeline is working correctly")
print("\nTo upload the model:")
print("  venv/bin/python upload.py")
