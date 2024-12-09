import os
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables dari file .env
load_dotenv()

# Ambil nama bucket dan nama project dari environment
bucket_name = os.getenv("GCP_BUCKET_NAME")
project_name = os.getenv("GCP_PROJECT_NAME")

# Pastikan nama bucket valid
if not bucket_name:
    raise ValueError("Nama bucket tidak ditemukan atau tidak valid!")

# Inisialisasi klien Google Cloud Storage
try:
    storage_client = storage.Client(project=project_name)
    bucket = storage_client.bucket(bucket_name)
    
except Exception as e:
    print(f"Terjadi kesalahan saat mengakses bucket: {e}")

