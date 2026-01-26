"""
Convert pickle model files to joblib format
"""
import pickle
import joblib

print("Converting models from pickle to joblib format...")

# Convert sentiment model
print("\n[1/2] Converting sentiment_model.pkl...")
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
joblib.dump(model, 'sentiment_model.joblib')
print("✓ Saved as sentiment_model.joblib")

# Convert TF-IDF vectorizer
print("\n[2/2] Converting tfidf_vectorizer.pkl...")
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
print("✓ Saved as tfidf_vectorizer.joblib")

print("\n" + "="*70)
print("Conversion completed successfully!")
print("="*70)

# Verify by loading
print("\nVerifying joblib files...")
loaded_model = joblib.load('sentiment_model.joblib')
loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')
print("✓ Both files loaded successfully")

# Show file sizes
import os
pkl_model_size = os.path.getsize('sentiment_model.pkl') / 1024
joblib_model_size = os.path.getsize('sentiment_model.joblib') / 1024
pkl_vec_size = os.path.getsize('tfidf_vectorizer.pkl') / 1024
joblib_vec_size = os.path.getsize('tfidf_vectorizer.joblib') / 1024

print(f"\nFile size comparison:")
print(f"  sentiment_model.pkl:      {pkl_model_size:.2f} KB")
print(f"  sentiment_model.joblib:   {joblib_model_size:.2f} KB")
print(f"  tfidf_vectorizer.pkl:     {pkl_vec_size:.2f} KB")
print(f"  tfidf_vectorizer.joblib:  {joblib_vec_size:.2f} KB")
