import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

class Message:
    def __init__(self):
        self.message = {
            "disease": {},
            "status_code": 0,
            "iscorn": True,
            "predicted-image":"",
            "timestamp": self._get_current_timestamp()
        }
    def _get_current_timestamp(self):
        return datetime.now().isoformat()

    def add_disease(self, key):
        if not isinstance(key, str):
            raise ValueError("Disease key must be a string")
        if key in self.message["disease"]:
            logging.warning(f"Disease '{key}' already exists, skipping.")
            return

        self.message["disease"][key] = {}
        logging.info(f"Disease '{key}' added successfully.")

    def add_section(self, key, section, content):
        if key not in self.message["disease"]:
            raise KeyError(f"Disease '{key}' does not exist. Add it first.")
        if not isinstance(section, str) or not isinstance(content, str):
            raise ValueError("Section and content must be strings")
        self.message["disease"][key][section] = content
        logging.info(f"Section '{section}' added to disease '{key}'.")
        logging.info(f"{self.get_message()['disease'][key]}")

    def set_status_code(self, status_code, posisi="Undefined"):
        if not isinstance(status_code, int):
            raise ValueError("Status code must be an integer")

        self.message['status_code'] = status_code
        logging.info(f"Status code set to {status_code}. {posisi}")

    def set_image(self, image):
        self.message['predicted-image'] = image
        logging.info(f"Sucessfull adding image")
    def get_message(self):
        return self.message

    def to_json(self):
        try:
            return json.dumps(self.get_message)
        except TypeError as e:
            logging.error("Error serializing message to JSON: ", exc_info=True)
            raise ValueError(f"Serialization error: {str(e)}")

    def from_json(self, json_str):
        try:
            parsed_message = json.loads(json_str)
            if "disease" in parsed_message and "status_code" in parsed_message:
                self.message = parsed_message
                logging.info("Message loaded from JSON successfully.")
            else:
                raise ValueError("Invalid JSON structure")
        except json.JSONDecodeError as e:
            logging.error("Error parsing JSON: ", exc_info=True)
            raise ValueError(f"Deserialization error: {str(e)}")
