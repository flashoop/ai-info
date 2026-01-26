import joblib

print("Testing sentiment_model.joblib...")
print("="*70)

# Load model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

print("✓ Model loaded successfully")
print(f"  Type: {type(model).__name__}")
print(f"  Features: {model.coef_.shape[1]:,}")

print("✓ Vectorizer loaded successfully")
print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

# Quick test
test_text = "This movie is amazing!"
features = vectorizer.transform([test_text])
prediction = model.predict(features)[0]
proba = model.predict_proba(features)[0][prediction]

print(f"\nQuick Test:")
print(f"  Input: '{test_text}'")
print(f"  Prediction: {'Positive' if prediction == 1 else 'Negative'}")
print(f"  Confidence: {proba*100:.2f}%")

print("\n" + "="*70)
print("✓ Model is ready for upload!")
