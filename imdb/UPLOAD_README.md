# Model Upload Guide

This guide explains how to upload the trained sentiment analysis model to your API server with progress tracking.

## Model Review

**File:** `sentiment_model.joblib`
- **Size:** 79 KB (80,863 bytes)
- **Format:** Joblib (scikit-learn standard)
- **Model Type:** Logistic Regression
- **Performance:** 88% accuracy on IMDB test set
- **Features:** Trained on TF-IDF features (10,000 features, unigrams + bigrams)

**Companion File:** `tfidf_vectorizer.joblib`
- **Size:** 368 KB
- **Purpose:** Text preprocessing and feature extraction
- **Note:** Both files are needed for inference

## Upload Scripts

We provide 3 different upload scripts with varying complexity:

### 1. upload.py (Recommended - With tqdm)

**Features:**
- Beautiful progress bar using `tqdm`
- Shows upload speed and ETA
- Color-coded progress
- Error handling

**Usage:**
```bash
cd /home/leo/ai-info/imdb
venv/bin/python upload.py
```

**Requirements:**
```bash
pip install tqdm requests
```

**Output Example:**
```
======================================================================
Uploading Model to API
======================================================================
File: sentiment_model.joblib
Size: 0.08 MB (80,863 bytes)
Endpoint: http://10.129.9.118:5000/api/upload
======================================================================
Uploading: 100%|████████████████████| 79.0k/79.0k [00:01<00:00, 45.2kB/s]

======================================================================
Upload Complete!
======================================================================
```

### 2. upload_simple.py (No External Dependencies)

**Features:**
- Simple progress bar without external dependencies
- Shows percentage and upload speed
- Only requires `requests` library

**Usage:**
```bash
venv/bin/python upload_simple.py
```

**Output Example:**
```
[████████████████████████████████████████] 100.0% | 1024.5 KB/s
```

### 3. upload_demo.py (Test/Demo)

**Features:**
- Demonstrates progress tracking without uploading
- Safe to run for testing
- Shows how the progress bar will look

**Usage:**
```bash
venv/bin/python upload_demo.py
```

## Configuration

Edit the upload script to configure:

```python
# API endpoint
url = "http://10.129.9.118:5000/api/upload"

# Model file to upload
model_file_path = "sentiment_model.joblib"
```

## API Endpoint Requirements

Your API endpoint should:
1. Accept POST requests
2. Accept file uploads with field name "model"
3. Return JSON response with upload status

**Example Flask Endpoint:**
```python
@app.route('/api/upload', methods=['POST'])
def upload_model():
    if 'model' not in request.files:
        return jsonify({'error': 'No model file'}), 400

    file = request.files['model']
    file.save('uploaded_model.joblib')

    return jsonify({
        'status': 'success',
        'message': 'Model uploaded successfully',
        'filename': file.filename
    })
```

## Error Handling

All scripts include error handling for:
- **File Not Found:** Model file doesn't exist
- **Connection Error:** Cannot reach the server
- **HTTP Errors:** Server returns error status code

## Troubleshooting

### Connection Refused
```
✗ Error: Could not connect to http://10.129.9.118:5000/api/upload
```
**Solution:** Ensure the API server is running and accessible.

### File Not Found
```
✗ Error: File 'sentiment_model.joblib' not found!
```
**Solution:** Run the script from the `/home/leo/ai-info/imdb` directory.

### Module Not Found (tqdm)
```
ModuleNotFoundError: No module named 'tqdm'
```
**Solution:** Install dependencies:
```bash
venv/bin/pip install tqdm requests
```

Or use `upload_simple.py` which doesn't require tqdm.

## Performance

**Model Size:** 79 KB
**Typical Upload Time:**
- Local network: < 1 second
- Internet (1 Mbps): ~1 second
- Internet (10 Mbps): < 0.5 seconds

## Security Notes

⚠️ **Important:**
- The model file contains trained weights only (no sensitive data)
- Ensure your API endpoint uses HTTPS in production
- Consider adding authentication to your upload endpoint
- Validate file size and type on the server side

## Next Steps

After uploading the model:
1. Load it on the server: `joblib.load('uploaded_model.joblib')`
2. Load the vectorizer: `joblib.load('tfidf_vectorizer.joblib')`
3. Create prediction endpoint
4. Test with sample movie reviews

## Example Prediction Code

```python
import joblib
import re

# Load model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

def predict_sentiment(text):
    # Clean text
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Transform and predict
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        'sentiment': 'positive' if prediction == 1 else 'negative',
        'confidence': float(probability[prediction])
    }
```
