"""
Example script: Load model from joblib and make predictions
"""
import joblib
import re

# Load model and vectorizer from joblib files
print("Loading model and vectorizer from joblib files...")
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
print("✓ Loaded successfully!\n")

def clean_text(text):
    """Clean text by removing HTML tags and extra whitespace"""
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text):
    """Predict sentiment for a given text"""
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[prediction] * 100
    return sentiment, confidence

# Test examples
if __name__ == "__main__":
    examples = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible movie. Waste of time and money. Don't watch it.",
        "It was okay, nothing special but not terrible either.",
        "One of the best films I've ever seen. Masterpiece!",
        "Boring and predictable. The acting was awful."
    ]

    print("Testing sentiment predictions:\n")
    print("="*70)
    for text in examples:
        sentiment, confidence = predict_sentiment(text)
        print(f"Text: {text}")
        print(f"→ {sentiment} (Confidence: {confidence:.2f}%)\n")
        print("-"*70)
