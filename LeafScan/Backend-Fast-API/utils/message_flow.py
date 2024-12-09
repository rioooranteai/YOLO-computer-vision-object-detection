import logging
import multiprocessing
from llm.text_generation import TextGeneration
from utils.create_message import Message


class Flow:
    def __init__(self):
        # Hindari penggunaan Manager.dict() di dalam Pool proses
        self.message = Message()
        self.text_generation = TextGeneration()

    def create_content(self, disease: str, section: str):
        try:
            # Menghasilkan teks menggunakan model LLM
            content = self.text_generation.generate_text(disease, section)

            # Membuat dictionary lokal dengan hasil untuk dikembalikan ke proses utama
            result = {disease: {section: content}}
            logging.info(f"Content created for {disease}-{section}")
            return result
        except Exception as e:
            logging.error(f"Error while creating content for disease '{disease}': {e}", exc_info=True)
            return None

    def _process_prediction(self, prediction, sections):
        results = {}
        for section in sections:
            content = self.create_content(prediction, section)
            if content:
                # Gabungkan hasil konten ke dalam dictionary lokal
                if prediction not in results:
                    results[prediction] = {}
                results[prediction].update(content[prediction])
        return results

    def result_flow(self, predictions: list, image):
        sections = ["Definisi","Diagnosa", "Saran"]

        try:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                # Gunakan Pool dan kumpulkan hasil proses dari tiap prediksi
                results = pool.starmap(self._process_prediction, [(prediction, sections) for prediction in predictions])

            # Gabungkan semua hasil ke dalam self.message di proses utama
            for result in results:
                if result:
                    for disease, content in result.items():
                        if disease not in self.message.get_message()["disease"]:
                            self.message.add_disease(disease)
                        for section, text in content.items():
                            self.message.add_section(disease, section, text)

            # Set status code berdasarkan hasil
            if all(results):
                self.message.set_status_code(200)
            else:
                self.message.set_status_code(500)

            logging.info(f"Final message: Berhasil")

            self.message.set_image(image)

            return self.message.get_message()
        except Exception as e:
            logging.error("An error occurred in result_flow: %s", e, exc_info=True)
            self.message.set_status_code(500)
            return self.message.get_message()

