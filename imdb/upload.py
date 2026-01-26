import requests
import json
import os
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper

# Define the URL of the API endpoint
url = "http://10.129.9.208:5000/api/upload"

# Path to the model file you want to upload
model_file_path = "skills_assessment.joblib"

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

# Open the file in binary mode with progress tracking
try:
    with open(model_file_path, "rb") as model_file:
        # Create progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024,
                  desc="Uploading", ncols=80) as progress_bar:

            # Wrap file with progress callback
            wrapped_file = CallbackIOWrapper(progress_bar.update, model_file, "read")

            files = {"model": wrapped_file}
            response = requests.post(url, files=files)

        print("\n" + "="*70)
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