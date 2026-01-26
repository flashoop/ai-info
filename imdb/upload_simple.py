"""
Simple upload script with basic progress tracking (no tqdm dependency)
"""
import requests
import json
import os
import sys
import time

# Define the URL of the API endpoint
url = "http://10.129.9.118:5000/api/upload"

# Path to the model file you want to upload
model_file_path = "sentiment_model.joblib"

# Get file size
file_size = os.path.getsize(model_file_path)
file_size_mb = file_size / (1024 * 1024)

print("="*70)
print(f"Uploading Model to API")
print("="*70)
print(f"File: {model_file_path}")
print(f"Size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
print(f"Endpoint: {url}")
print("="*70)

class ProgressFile:
    """File wrapper that tracks upload progress"""
    def __init__(self, file_obj, file_size):
        self.file_obj = file_obj
        self.file_size = file_size
        self.bytes_read = 0
        self.start_time = time.time()

    def read(self, size=-1):
        chunk = self.file_obj.read(size)
        self.bytes_read += len(chunk)

        # Calculate progress
        progress = (self.bytes_read / self.file_size) * 100
        elapsed = time.time() - self.start_time

        if elapsed > 0:
            speed = self.bytes_read / elapsed / 1024  # KB/s

            # Print progress bar
            bar_length = 40
            filled = int(bar_length * self.bytes_read / self.file_size)
            bar = '█' * filled + '░' * (bar_length - filled)

            sys.stdout.write(f'\r[{bar}] {progress:.1f}% | {speed:.1f} KB/s')
            sys.stdout.flush()

        return chunk

    def __len__(self):
        return self.file_size

# Open the file in binary mode with progress tracking
try:
    print("\nUploading...")
    with open(model_file_path, "rb") as model_file:
        progress_file = ProgressFile(model_file, file_size)
        files = {"model": progress_file}
        response = requests.post(url, files=files)

    print("\n\n" + "="*70)
    print("Upload Complete!")
    print("="*70)

    # Pretty print the response from the server
    if response.status_code == 200:
        print("\n✓ Success! Server Response:")
        print(json.dumps(response.json(), indent=4))
    else:
        print(f"\n✗ Error! Status Code: {response.status_code}")
        print("Response:")
        print(response.text)

except FileNotFoundError:
    print(f"\n✗ Error: File '{model_file_path}' not found!")
except requests.exceptions.ConnectionError:
    print(f"\n✗ Error: Could not connect to {url}")
    print("Please check if the server is running and the URL is correct.")
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
