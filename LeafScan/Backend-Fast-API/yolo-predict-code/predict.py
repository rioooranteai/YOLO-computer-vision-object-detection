from google.cloud import storage
import cv2
import numpy as np
from ultralytics import YOLO
import os  # Import os to handle file paths

from . import storage_client, bucket_name, model


def upload_image_to_gcp(image, destination_blob_name):
    """
    Function to upload an image to Google Cloud Storage.
    """
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    _, image_encoded = cv2.imencode('.jpg', image)
    blob.upload_from_string(image_encoded.tobytes(), content_type='image/jpeg')

    return f"gs://{bucket_name}/{destination_blob_name}"


def save_image_locally(image, local_path):
    """
    Function to save an image to the local filesystem.
    """
    cv2.imwrite(local_path, image)


def predict_image_obj_detection(image_file, confidence_level=0.5):
    """
    Function to predict an uploaded image using the YOLO model.
    The image is uploaded to GCP first, and the prediction results are also uploaded to GCP.

    :param image_file: The uploaded image file.
    :param confidence_level: The confidence threshold for the detection model.
    :return: A dictionary containing the original image URL, detected classes, and local predicted image path.
    """
    # Membaca gambar dari file yang diunggah
    image_array = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Simpan gambar yang diunggah ke direktori lokal
    uploaded_image_path = os.path.join("D:\Computer-Vision\LeafScan\predicted", f"uploaded_{image_file.filename}")
    cv2.imwrite(uploaded_image_path, image)
    print(f"Uploaded image saved to: {uploaded_image_path}")  # Untuk debugging

    # Nama blob untuk GCP
    source_blob_name = f"uploaded_images/{image_file.filename}"

    # Melakukan deteksi objek dengan model YOLO
    results_1 = model(image, conf=confidence_level)

    detection = set()
    for result in results_1:
        if result.names:
            detected_classes = [result.names[int(label)] for label in result.boxes.cls.cpu().numpy()]
            detection.update(detected_classes)

    detection = list(detection)

    # Pastikan direktori untuk menyimpan gambar prediksi ada
    os.makedirs("D:\Computer-Vision\LeafScan\predicted", exist_ok=True)

    # Menyimpan gambar hasil deteksi
    for i, result in enumerate(results_1):
        saved_path = os.path.join("D:\Computer-Vision\LeafScan\predicted", f"predicted_{i}.jpg")
        result.save(saved_path)
        print(f"Saved image to: {saved_path}")  # Untuk debugging

    return {
        "original_image_url": f"gs://{bucket_name}/{source_blob_name}",
        "detected_classes": detection,
        "local_predicted_image_path":"D:\Computer-Vision\LeafScan\predicted" }
