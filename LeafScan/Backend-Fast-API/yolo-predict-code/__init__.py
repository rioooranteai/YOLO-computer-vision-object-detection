import os
from dotenv import load_dotenv
from google.cloud import storage
from ultralytics import YOLO
from google.cloud import storage
import cv2
import numpy as np
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction


# Inisialisasi klien Google Cloud Storage
storage_client = storage.Client()

# Inisialisasi YOLO model
model_path = "models/best-v2-2510240737.pt"  # Path ke model YOLO
model = YOLO(model_path)
