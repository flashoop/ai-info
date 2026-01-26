"""
IMDB Sentiment Classification - Option A: Logistic Regression with TF-IDF
Goal: Predict whether a movie review is positive (1) or negative (0)
"""

import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("IMDB Sentiment Classification - Logistic Regression + TF-IDF")
print("="*70)

# ============================================================================
# 1. Load Data
# ============================================================================
print("\n[1/8] Loading data...")
with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"✓ Training samples: {len(train_data)}")
print(f"✓ Test samples: {len(test_data)}")

# ============================================================================
# 2. Explore Data
# ============================================================================
print("\n[2/8] Exploring data...")
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print(f"✓ Label distribution: {dict(train_df['label'].value_counts())}")
train_df['text_length'] = train_df['text'].apply(len)
train_df['word_count'] = train_df['text'].apply(lambda x: len(x.split()))
print(f"✓ Avg words per review: {train_df['word_count'].mean():.0f}")

# ============================================================================
# 3. Text Preprocessing
# ============================================================================
print("\n[3/8] Preprocessing text...")

def clean_text(text):
    """Clean text by removing HTML tags and extra whitespace"""
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)
print("✓ Text cleaning completed")

# ============================================================================
# 4. Split Training Data
# ============================================================================
print("\n[4/8] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    train_df['clean_text'],
    train_df['label'],
    test_size=0.2,
    random_state=42,
    stratify=train_df['label']
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Validation set: {len(X_val)} samples")
print(f"✓ Test set: {len(test_df)} samples")

# ============================================================================
# 5. TF-IDF Vectorization
# ============================================================================
print("\n[5/8] Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(test_df['clean_text'])

print(f"✓ TF-IDF matrix shape: {X_train_tfidf.shape}")
print(f"✓ Number of features: {len(tfidf.get_feature_names_out())}")

# ============================================================================
# 6. Train Logistic Regression Model
# ============================================================================
print("\n[6/8] Training Logistic Regression model...")
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train_tfidf, y_train)
print("✓ Training completed!")

# ============================================================================
# 7. Evaluate on Validation Set
# ============================================================================
print("\n[7/8] Evaluating on validation set...")
y_val_pred = model.predict(X_val_tfidf)
y_val_proba = model.predict_proba(X_val_tfidf)[:, 1]

val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

print("\n" + "="*70)
print("VALIDATION SET RESULTS")
print("="*70)
print(f"Accuracy:  {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall:    {val_recall:.4f}")
print(f"F1-Score:  {val_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Validation Set - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_val.png', dpi=150)
print("✓ Validation confusion matrix saved to confusion_matrix_val.png")

# ============================================================================
# 8. Evaluate on Test Set
# ============================================================================
print("\n[8/8] Evaluating on test set...")
y_test_pred = model.predict(X_test_tfidf)
y_test_proba = model.predict_proba(X_test_tfidf)[:, 1]

test_accuracy = accuracy_score(test_df['label'], y_test_pred)
test_precision = precision_score(test_df['label'], y_test_pred)
test_recall = recall_score(test_df['label'], y_test_pred)
test_f1 = f1_score(test_df['label'], y_test_pred)

print("\n" + "="*70)
print("TEST SET RESULTS")
print("="*70)
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(test_df['label'], y_test_pred, target_names=['Negative', 'Positive']))

# Confusion Matrix for test set
cm_test = confusion_matrix(test_df['label'], y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Test Set - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_test.png', dpi=150)
print("✓ Test confusion matrix saved to confusion_matrix_test.png")

# ============================================================================
# Analyze Most Important Features
# ============================================================================
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

top_positive_idx = np.argsort(coefficients)[-20:]
top_positive = [(feature_names[i], coefficients[i]) for i in top_positive_idx]

top_negative_idx = np.argsort(coefficients)[:20]
top_negative = [(feature_names[i], coefficients[i]) for i in top_negative_idx]

print("\nTop 20 words indicating POSITIVE sentiment:")
for word, coef in reversed(top_positive):
    print(f"  {word:25s} {coef:8.4f}")

print("\nTop 20 words indicating NEGATIVE sentiment:")
for word, coef in top_negative:
    print(f"  {word:25s} {coef:8.4f}")

# ============================================================================
# Test on Custom Examples
# ============================================================================
print("\n" + "="*70)
print("TESTING ON CUSTOM EXAMPLES")
print("="*70)

def predict_sentiment(text):
    """Predict sentiment for a given text"""
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[prediction] * 100
    return sentiment, confidence

examples = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible movie. Waste of time and money. Don't watch it.",
    "It was okay, nothing special but not terrible either.",
    "One of the best films I've ever seen. Masterpiece!",
    "Boring and predictable. The acting was awful."
]

for text in examples:
    sentiment, confidence = predict_sentiment(text)
    print(f"\nText: {text}")
    print(f"→ Prediction: {sentiment} (Confidence: {confidence:.2f}%)")

# ============================================================================
# Save Model and Artifacts
# ============================================================================
print("\n" + "="*70)
print("SAVING MODEL AND ARTIFACTS")
print("="*70)

joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

results = {
    'test_accuracy': float(test_accuracy),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'test_f1': float(test_f1),
    'val_accuracy': float(val_accuracy),
    'val_precision': float(val_precision),
    'val_recall': float(val_recall),
    'val_f1': float(val_f1)
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Model saved: sentiment_model.joblib")
print("✓ Vectorizer saved: tfidf_vectorizer.joblib")
print("✓ Results saved: results.json")

print("\n" + "="*70)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
