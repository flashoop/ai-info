# Skills Assessment Model - IMDB Sentiment Classification

## Overview

This project creates an IMDB sentiment classification model following the same structure and methodology as the SMS spam classification project. The model predicts whether a movie review is **positive (1)** or **negative (0)**.

## Model Information

**File:** `skills_assessment.joblib`
- **Size:** 445 KB (455,236 bytes)
- **Algorithm:** Logistic Regression with TF-IDF Vectorization
- **Features:** 10,000 TF-IDF features (unigrams + bigrams)
- **Best Hyperparameters:** C=2.0, penalty='l2'

## Performance Metrics

### Cross-Validation (5-Fold)
- **F1-Score:** 88.85%

### Validation Set (4,981 samples)
- **Accuracy:** 89.02%
- **Precision:** 87.72%
- **Recall:** 90.78%
- **F1-Score:** 89.23%

### Test Set (25,000 samples)
- **Accuracy:** 88.29%
- **Precision:** 88.11%
- **Recall:** 88.53%
- **F1-Score:** 88.32%

## Project Structure

Following the SMS spam classification approach:

```
imdb/
├── train.json                          # Training data (25,000 reviews)
├── test.json                           # Test data (25,000 reviews)
├── train_model.ipynb                   # Interactive Jupyter notebook (complete pipeline)
├── train_skills_assessment.py          # Python script version
├── skills_assessment.joblib            # Trained model (MAIN OUTPUT)
├── skills_assessment_results.json      # Performance metrics
├── verify_skills_assessment.py         # Model verification script
├── confusion_matrix_validation.png     # Validation confusion matrix
├── confusion_matrix_test_skills.png    # Test confusion matrix
└── training_log.txt                    # Complete training log
```

## Training Pipeline

The model follows the exact same structure as the SMS spam classification:

### 1. **Data Loading & Exploration**
- Loaded 25,000 training and 25,000 test reviews
- Checked for missing values and duplicates
- Analyzed label distribution (balanced: 50% positive, 50% negative)

### 2. **Preprocessing Pipeline**
Similar to SMS spam preprocessing:

- **Lowercasing:** Convert all text to lowercase
- **HTML Removal:** Remove `<br />` tags and other HTML elements
- **Cleaning:** Remove punctuation and numbers
- **Tokenization:** Split text into individual words
- **Stop Word Removal:** Remove common words (the, is, and, etc.)
- **Stemming:** Reduce words to root form (Porter Stemmer)

Example:
```
Original: "This movie was <br /> absolutely fantastic!"
After Cleaning: "this movie was absolutely fantastic"
After Processing: "movi absolut fantast"
```

### 3. **Feature Extraction**
- **Method:** TF-IDF Vectorization
- **Parameters:**
  - max_features=10,000
  - ngram_range=(1, 2) - unigrams and bigrams
  - min_df=5 - minimum document frequency
  - max_df=0.8 - maximum document frequency

### 4. **Model Training**
- **Algorithm:** Logistic Regression
- **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
- **Parameter Grid:** C in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
- **Best Parameters:** C=2.0

### 5. **Evaluation**
- Comprehensive metrics on validation and test sets
- Confusion matrices for visual analysis
- Feature importance analysis

## Key Features Learned

### Top Positive Indicators:
- excel (6.62), great (6.50), perfect (5.38)
- love (4.88), enjoy (4.65), favorit (4.55)
- best (4.39), amaz (4.11), fun (4.07)

### Top Negative Indicators:
- worst (-8.97), aw (-7.81), wast (-7.21)
- bad (-6.86), bore (-6.79), poor (-5.53)
- disappoint (-5.39), fail (-5.36), dull (-5.35)

## Usage

### Loading the Model

```python
import joblib

# Load model
model = joblib.load('skills_assessment.joblib')
```

### Making Predictions

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize preprocessing tools
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
    text = text.lower()
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Make prediction
review = "This movie was absolutely fantastic! I loved it!"
processed_review = preprocess_text(review)
prediction = model.predict([processed_review])[0]
probabilities = model.predict_proba([processed_review])[0]

sentiment = "Positive" if prediction == 1 else "Negative"
confidence = probabilities[prediction] * 100

print(f"Sentiment: {sentiment} ({confidence:.2f}% confidence)")
```

### Verification Script

```bash
cd /home/leo/ai-info/imdb
venv/bin/python verify_skills_assessment.py
```

## Comparison with SMS Spam Classification

| Aspect | SMS Spam | IMDB Sentiment |
|--------|----------|----------------|
| **Algorithm** | MultinomialNB | Logistic Regression |
| **Vectorization** | CountVectorizer (BoW) | TfidfVectorizer |
| **Features** | Bigrams | Unigrams + Bigrams |
| **Dataset Size** | 5,572 messages | 50,000 reviews |
| **Best Param** | alpha=0.5 | C=2.0 |
| **Preprocessing** | Same pipeline | Same pipeline |
| **Evaluation** | GridSearchCV + F1 | GridSearchCV + F1 |
| **Output Format** | joblib | joblib |

## Files Generated

1. **train_model.ipynb** - Jupyter notebook with complete pipeline (similar to SMS spam notebooks)
2. **train_skills_assessment.py** - Python script version for automated training
3. **skills_assessment.joblib** - Main trained model file
4. **skills_assessment_results.json** - Performance metrics in JSON format
5. **confusion_matrix_validation.png** - Validation confusion matrix
6. **confusion_matrix_test_skills.png** - Test confusion matrix
7. **training_log.txt** - Complete training output log
8. **verify_skills_assessment.py** - Model verification script

## Training Commands

### Interactive Notebook
```bash
cd /home/leo/ai-info/imdb
jupyter notebook train_model.ipynb
```

### Python Script
```bash
cd /home/leo/ai-info/imdb
venv/bin/python train_skills_assessment.py
```

## Model Performance Summary

The model achieves **88.29% accuracy** on the test set, demonstrating strong performance in binary sentiment classification. The balanced precision (88.11%) and recall (88.53%) indicate the model performs equally well on both positive and negative reviews.

### Classification Report (Test Set):
```
              precision    recall  f1-score   support

    Negative       0.88      0.88      0.88     12500
    Positive       0.88      0.89      0.88     12500

    accuracy                           0.88     25000
```

## Upload to API

To upload the model to your API endpoint:

```bash
cd /home/leo/ai-info/imdb

# Update upload.py to use skills_assessment.joblib
# Then run:
venv/bin/python upload.py
```

Edit `upload.py` line 11:
```python
model_file_path = "skills_assessment.joblib"
```

## Requirements

The model was trained with:
- Python 3.12
- scikit-learn 1.8.0
- pandas 3.0.0
- numpy 2.4.1
- nltk 3.9.2
- matplotlib 3.10.8
- seaborn 0.13.2
- joblib 1.5.3

Install all requirements:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk joblib
```

## Notes

- The model uses the same preprocessing methodology as the SMS spam classifier
- NLTK data (punkt, stopwords) must be downloaded before use
- Preprocessing must be applied identically during training and inference
- The model is saved as a complete pipeline including both vectorizer and classifier

---

**Created:** 2026-01-26
**Status:** ✓ Complete and Verified
**Model File:** `skills_assessment.joblib`
