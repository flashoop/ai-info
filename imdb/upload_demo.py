"""
Demo script to show upload progress without actually uploading to a server
This simulates reading the file with progress tracking
"""
import os
import time
from tqdm import tqdm

# Path to the model file
model_file_path = "sentiment_model.joblib"

# Get file size
file_size = os.path.getsize(model_file_path)
file_size_mb = file_size / (1024 * 1024)

print("="*70)
print(f"Upload Progress Demo")
print("="*70)
print(f"File: {model_file_path}")
print(f"Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
print("="*70)
print("\nSimulating upload with progress bar...\n")

# Simulate upload with progress bar
chunk_size = 8192  # 8KB chunks
with open(model_file_path, "rb") as f:
    with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024,
              desc="Uploading", ncols=80, colour='green') as pbar:

        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # Update progress
            pbar.update(len(chunk))

            # Simulate network delay (remove this in actual upload)
            time.sleep(0.01)

print("\n" + "="*70)
print("✓ Demo Complete!")
print("="*70)
print("\nThis is how the progress bar will look during actual upload.")
print("To upload to your server, use: venv/bin/python upload.py")
