"""
Verify skills_assessment.joblib model
"""
import joblib
import os
import json

print("=" * 70)
print("SKILLS ASSESSMENT MODEL VERIFICATION")
print("=" * 70)

# Check file exists
model_file = 'skills_assessment.joblib'
if not os.path.exists(model_file):
    print(f"✗ Error: {model_file} not found!")
    exit(1)

# Load model
print(f"\n[1] Loading model from {model_file}...")
model = joblib.load(model_file)
print(f"✓ Model loaded successfully")
print(f"  Type: {type(model).__name__}")

# Get file info
file_size = os.path.getsize(model_file)
print(f"  Size: {file_size / 1024:.2f} KB ({file_size:,} bytes)")

# Check model components
print(f"\n[2] Model Components:")
if hasattr(model, 'named_steps'):
    for step_name in model.named_steps:
        step = model.named_steps[step_name]
        print(f"  - {step_name}: {type(step).__name__}")

# Check vectorizer
vectorizer = model.named_steps['vectorizer']
print(f"\n[3] Vectorizer Configuration:")
print(f"  Max features: {vectorizer.max_features}")
print(f"  Ngram range: {vectorizer.ngram_range}")
print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

# Check classifier
classifier = model.named_steps['classifier']
print(f"\n[4] Classifier Configuration:")
if hasattr(classifier, 'C'):
    print(f"  Regularization (C): {classifier.C}")
if hasattr(classifier, 'coef_'):
    print(f"  Number of features: {classifier.coef_.shape[1]:,}")
    print(f"  Classes: {classifier.classes_.tolist()}")

# Load results
print(f"\n[5] Performance Metrics:")
with open('skills_assessment_results.json', 'r') as f:
    results = json.load(f)

print(f"  Best CV F1: {results['best_cv_f1_score']:.4f}")
print(f"  Validation Accuracy: {results['validation']['accuracy']:.4f}")
print(f"  Test Accuracy: {results['test']['accuracy']:.4f}")
print(f"  Test F1-Score: {results['test']['f1_score']:.4f}")

# Quick prediction test
print(f"\n[6] Quick Prediction Test:")

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = text.lower()
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

test_cases = [
    ("Amazing movie! Loved it!", "Positive"),
    ("Terrible and boring waste of time.", "Negative")
]

all_correct = True
for text, expected in test_cases:
    processed = preprocess_text(text)
    prediction = model.predict([processed])[0]
    proba = model.predict_proba([processed])[0]
    predicted_label = "Positive" if prediction == 1 else "Negative"
    confidence = proba[prediction] * 100

    match = "✓" if predicted_label == expected else "✗"
    print(f"\n  {match} Text: {text}")
    print(f"    Expected: {expected}, Got: {predicted_label} ({confidence:.2f}%)")

    if predicted_label != expected:
        all_correct = False

print("\n" + "=" * 70)
if all_correct:
    print("✓ ALL VERIFICATIONS PASSED - MODEL IS READY!")
else:
    print("✗ SOME VERIFICATIONS FAILED - CHECK MODEL")
print("=" * 70)
