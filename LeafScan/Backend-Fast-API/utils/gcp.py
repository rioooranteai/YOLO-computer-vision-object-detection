import json
import cv2
import sys
import datetime
import os

from . import bucket

REPORT = {}

def upload_results_to_bucket(folder, data):
    try:
        # Membuat nama file dan destination_path
        file_name = f"detect-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        destination_path = f"history/{folder}/{file_name}"

        print(f"Uploading results to bucket {destination_path}")

        # Mengunggah ke bucket
        blob = bucket.blob(destination_path)
        blob.upload_from_string(json.dumps(data), content_type='application/json')

    except Exception as e:
        error_message = f"Error uploading results to bucket {destination_path}: {e}"
        REPORT['error'] = error_message
        print(error_message)
        raise
