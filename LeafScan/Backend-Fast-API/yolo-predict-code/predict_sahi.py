import logging
import cv2
import os
import numpy as np
import base64
import shutil

from io import BytesIO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from google.cloud import storage
from . import model_path


def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def resize_image_by_percentage(image_bytes, scale_percent=100):

    image_array = np.frombuffer(image_bytes.getvalue(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    original_height, original_width = image.shape[:2]
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)
    dim = (new_width, new_height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized_image

def save_image_locally(image, local_path):
    cv2.imwrite(local_path, image)

def predict_image_obj_detection_sahi(image_bytes, image_name, confidence_level=0.5):

    image = resize_image_by_percentage(image_bytes, 60)
    uploaded_image_path = os.path.join("predicted", "uploaded_image.jpg")
    os.makedirs(os.path.dirname(uploaded_image_path), exist_ok=True)
    cv2.imwrite(uploaded_image_path, image)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=confidence_level,
        device="cpu"
    )

    result = get_sliced_prediction(
        uploaded_image_path,
        detection_model,
        slice_height=356,
        slice_width=356,
        overlap_width_ratio=0.5,
        overlap_height_ratio=0.5,
    )

    detected_classes = {obj.category.name for obj in result.object_prediction_list}
    
    logging.info(detected_classes)

    predicted_dir = "predicted"
    os.makedirs(predicted_dir, exist_ok=True)
    predicted_image_path = os.path.join(predicted_dir, "predicted_sahi.png")
    result.export_visuals(export_dir=predicted_dir, file_name="predicted_sahi")

    image_base64 = encode_image_to_base64(predicted_image_path)
    
    shutil.rmtree(predicted_dir)

    return {
        "original_image_url": "Test",
        "detected_classes": list(detected_classes),
        "local_predicted_image_path": predicted_image_path,
        "image": image_base64
    }

