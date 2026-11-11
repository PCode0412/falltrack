"""
SisFall Dataset Downloader and Extractor
Downloads the official SisFall dataset archive (~204 MB) and extracts it into the data/ directory.
"""
import os
import sys
import zipfile
import requests

DATASET_URL = "https://github.com/BIng2325/SisFall/releases/download/dataset/SisFall.zip"
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ZIP_PATH = os.path.join(TARGET_DIR, "SisFall.zip")

def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        print(f"File {dest_path} already exists ({size / (1024*1024):.2f} MB). Skipping download.")
        return
    
    print(f"Downloading SisFall dataset from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    downloaded = 0
    chunk_size = 1024 * 1024  # 1MB
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
                else:
                    print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
    print("\nDownload complete!")

def extract_zip(zip_path: str, extract_to: str):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

def main():
    download_file(DATASET_URL, ZIP_PATH)
    extract_zip(ZIP_PATH, TARGET_DIR)
    print(f"Dataset successfully prepared in {TARGET_DIR}")

if __name__ == "__main__":
    main()
