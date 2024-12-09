import logging
import uuid

from io import BytesIO
from fastapi import APIRouter, Request, UploadFile, File, HTTPException, Form
from utils.message_flow import Flow
from cv.predict_sahi import predict_image_obj_detection_sahi
from utils.gcp import upload_results_to_bucket


main = APIRouter()
# Buat instance Flow di luar fungsi untuk digunakan kembali
result_flow = Flow()

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

@main.post("/predict-image")
async def predict_image(image: UploadFile = File(...), username: str = Form(...)):
    """
    Route untuk menerima gambar dari method POST, memanggil fungsi prediksi,
    dan mengembalikan URL gambar asli, hasil prediksi dari GCP, serta kelas yang terdeteksi.
    """
    confidence_level = 0.3

    content_byte = await image.read()

    try:
        # Pastikan file gambar diterima dengan benar
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")

        # Mengonversi byte ke objek file menggunakan BytesIO
        image_file = BytesIO(content_byte)
        image_name = f"{uuid.uuid4()}"
        # Prediksi gambar dengan objek file BytesIO
        result = predict_image_obj_detection_sahi(image_file,image_name, confidence_level)

        # Pastikan result["detected_classes"] ada dan dalam format yang tepat
        if "detected_classes" not in result:
            logging.warning("No classes detected in the result")
            return {"error": "No classes detected"}, 400

        llm_result = result_flow.result_flow(result["detected_classes"], result["image"]) 

        return llm_result

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

