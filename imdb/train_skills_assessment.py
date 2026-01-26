"""
IMDB Sentiment Classification - Skills Assessment Model
Following the same structure as SMS spam classification project
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import joblib
import os

print("=" * 70)
print("IMDB SENTIMENT CLASSIFICATION - SKILLS ASSESSMENT")
print("=" * 70)

# ============================================================================
# Section 1: Load and Explore Dataset
# ============================================================================
print("\n[1/9] Loading dataset...")

with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open('test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print(f"✓ Training samples: {len(train_df)}")
print(f"✓ Test samples: {len(test_df)}")
print(f"✓ Label distribution: {dict(train_df['label'].value_counts())}")

# Remove duplicates if any
if train_df.duplicated().sum() > 0:
    train_df = train_df.drop_duplicates()
    print(f"✓ Removed {train_df.duplicated().sum()} duplicates")

# ============================================================================
# Section 2: Preprocessing Pipeline
# ============================================================================
print("\n[2/9] Preprocessing text...")

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """Remove HTML tags and special characters"""
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    """Complete preprocessing pipeline"""
    # Lowercase
    text = text.lower()
    # Clean
    text = clean_text(text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Join back
    return ' '.join(tokens)

# Apply preprocessing
print("  Preprocessing training data...")
train_df['processed_text'] = train_df['text'].apply(preprocess_text)

print("  Preprocessing test data...")
test_df['processed_text'] = test_df['text'].apply(preprocess_text)

print("✓ Preprocessing completed")

# ============================================================================
# Section 3: Split Data
# ============================================================================
print("\n[3/9] Splitting data...")

X_train, X_val, y_train, y_val = train_test_split(
    train_df['processed_text'],
    train_df['label'],
    test_size=0.2,
    random_state=42,
    stratify=train_df['label']
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Validation set: {len(X_val)} samples")
print(f"✓ Test set: {len(test_df)} samples")

# ============================================================================
# Section 4: Create Pipeline and Hyperparameter Tuning
# ============================================================================
print("\n[4/9] Creating pipeline and parameter grid...")

# Create pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8
    )),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Parameter grid
param_grid = {
    'classifier__C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    'classifier__penalty': ['l2']
}

print("✓ Pipeline created")
print(f"✓ Parameter grid: {param_grid}")

# ============================================================================
# Section 5: GridSearchCV Training
# ============================================================================
print("\n[5/9] Performing GridSearchCV (this may take several minutes)...")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

# Fit on training data
grid_search.fit(X_train, y_train)

# Extract best model
best_model = grid_search.best_estimator_

print("\n✓ Training completed!")
print(f"✓ Best parameters: {grid_search.best_params_}")
print(f"✓ Best F1 score (CV): {grid_search.best_score_:.4f}")

# ============================================================================
# Section 6: Validation Set Evaluation
# ============================================================================
print("\n[6/9] Evaluating on validation set...")

y_val_pred = best_model.predict(X_val)
y_val_proba = best_model.predict_proba(X_val)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

print("\n" + "=" * 70)
print("VALIDATION SET RESULTS")
print("=" * 70)
print(f"Accuracy:  {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall:    {val_recall:.4f}")
print(f"F1-Score:  {val_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Negative', 'Positive']))

# Confusion matrix
cm_val = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Validation Set - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_validation.png', dpi=150)
plt.close()
print("✓ Confusion matrix saved: confusion_matrix_validation.png")

# ============================================================================
# Section 7: Test Set Evaluation
# ============================================================================
print("\n[7/9] Evaluating on test set...")

y_test_pred = best_model.predict(test_df['processed_text'])
y_test_proba = best_model.predict_proba(test_df['processed_text'])

test_accuracy = accuracy_score(test_df['label'], y_test_pred)
test_precision = precision_score(test_df['label'], y_test_pred)
test_recall = recall_score(test_df['label'], y_test_pred)
test_f1 = f1_score(test_df['label'], y_test_pred)

print("\n" + "=" * 70)
print("TEST SET RESULTS")
print("=" * 70)
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")
print("\nClassification Report:")
print(classification_report(test_df['label'], y_test_pred, target_names=['Negative', 'Positive']))

# Confusion matrix
cm_test = confusion_matrix(test_df['label'], y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Test Set - Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_test_skills.png', dpi=150)
plt.close()
print("✓ Confusion matrix saved: confusion_matrix_test_skills.png")

# ============================================================================
# Section 8: Feature Importance Analysis
# ============================================================================
print("\n[8/9] Analyzing feature importance...")

vectorizer_from_pipeline = best_model.named_steps['vectorizer']
classifier_from_pipeline = best_model.named_steps['classifier']

feature_names = vectorizer_from_pipeline.get_feature_names_out()
coefficients = classifier_from_pipeline.coef_[0]

# Top positive features
top_positive_idx = np.argsort(coefficients)[-20:]
top_positive = [(feature_names[i], coefficients[i]) for i in top_positive_idx]

# Top negative features
top_negative_idx = np.argsort(coefficients)[:20]
top_negative = [(feature_names[i], coefficients[i]) for i in top_negative_idx]

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)
print("\nTop 20 words indicating POSITIVE sentiment:")
for word, coef in reversed(top_positive):
    print(f"  {word:25s} {coef:8.4f}")

print("\nTop 20 words indicating NEGATIVE sentiment:")
for word, coef in top_negative:
    print(f"  {word:25s} {coef:8.4f}")

# ============================================================================
# Section 9: Save Model and Results
# ============================================================================
print("\n[9/9] Saving model and results...")

# Save model
model_filename = 'skills_assessment.joblib'
joblib.dump(best_model, model_filename)

file_size = os.path.getsize(model_filename)
print(f"\n✓ Model saved: {model_filename}")
print(f"✓ File size: {file_size / 1024:.2f} KB ({file_size:,} bytes)")

# Save results to JSON
results = {
    'model_name': 'IMDB Sentiment Classification - Skills Assessment',
    'algorithm': 'Logistic Regression with TF-IDF',
    'best_params': grid_search.best_params_,
    'best_cv_f1_score': float(grid_search.best_score_),
    'validation': {
        'accuracy': float(val_accuracy),
        'precision': float(val_precision),
        'recall': float(val_recall),
        'f1_score': float(val_f1)
    },
    'test': {
        'accuracy': float(test_accuracy),
        'precision': float(test_precision),
        'recall': float(test_recall),
        'f1_score': float(test_f1)
    }
}

with open('skills_assessment_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Results saved: skills_assessment_results.json")

# ============================================================================
# Test Custom Examples
# ============================================================================
print("\n" + "=" * 70)
print("TESTING ON CUSTOM EXAMPLES")
print("=" * 70)

example_reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "Terrible movie. Complete waste of time and money.",
    "It was okay, nothing special but not terrible either.",
    "A masterpiece! Highly recommended!",
    "Boring and predictable. Don't waste your time."
]

for i, review in enumerate(example_reviews, 1):
    processed = preprocess_text(review)
    prediction = best_model.predict([processed])[0]
    probabilities = best_model.predict_proba([processed])[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[prediction] * 100

    print(f"\nExample {i}:")
    print(f"  Review: {review}")
    print(f"  Prediction: {sentiment} (Confidence: {confidence:.2f}%)")

print("\n" + "=" * 70)
print("✓ SKILLS ASSESSMENT MODEL TRAINING COMPLETED!")
print("=" * 70)
print(f"\nModel saved as: {model_filename}")
print(f"Ready for deployment and assessment!")
