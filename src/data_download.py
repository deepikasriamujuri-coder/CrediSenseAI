import os
import urllib.request
import sys

def download_dataset():
    url = "https://raw.githubusercontent.com/PhilChodrow/ml-notes/main/data/credit-risk/credit_risk_dataset.csv"
    raw_dir = os.path.join("data", "raw")
    dest_path = os.path.join(raw_dir, "credit_risk_dataset.csv")
    
    # Ensure raw directory exists
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f"Downloading dataset from: {url}")
    print(f"Saving to: {dest_path}")
    
    try:
        # User agent header to avoid request blocking
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            
        print("Download successful!")
        file_size = os.path.getsize(dest_path)
        print(f"File Size: {file_size / (1024 * 1024):.2f} MB")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    download_dataset()
