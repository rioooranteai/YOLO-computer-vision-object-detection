# app/llm/gpt-recommendation.py
import logging
from . import client

template_testing = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia, eros nec tincidunt gravida, erat justo ullamcorper velit, eget pharetra felis nunc ac magna. Maecenas euismod elit at enim facilisis, eu auctor elit cursus. Integer feugiat nulla et nisi dapibus, vel interdum turpis feugiat. Nulla fringilla ligula vel efficitur varius. Aenean sagittis, ipsum nec fringilla tincidunt, felis libero suscipit turpis, id malesuada ante metus at sapien. Sed a eros ut risus malesuada feugiat ac eget est. Donec efficitur augue ac elit rhoncus, vitae fringilla leo gravida. Ut ac odio at metus volutpat sollicitudin. Fusce et purus augue. Cras consectetur augue a auctor efficitur. Sed lobortis eros sit amet est tempus gravida. Vivamus a suscipit leo, et posuere justo.
"""

class TextGeneration:

    prompt_user = """
    Daun jagung terdeteksi memiliki disease {}. Tugas anda adalah membuat informasi tentang {} untuk
    konteks disease ini.Informasi yang anda berikan akan ditampilkan pada aplikasi deteksi penyakit jagung.
    Tujuan dari aplikasi ini adalah membatu user untuk memahami penyakit hingga mengatasi penyakit yang terdeteksi
    Sebagai edukator, anda harus memastikan penjelasan yang anda berikan jelas dan mudah dimengerti oleh
    pengguna. Hindari penggunaan markdown pada teks. 
    """

    prompt_system = "Kamu adalah seorang ahli dalam botani, khususnya tanaman jagung."
    


    def generate_text(self, disease, section):
        try:

            prompt = self.prompt_user.format(disease, section)
            
            completion = client.chat.completions.create(
                 model="gpt-3.5-turbo",
                 messages=[
                     {"role": "system", "content": self.prompt_system},
                     {"role": "user", "content": prompt}
                 ],
                 max_tokens=154,
             )

            response_text = completion.choices[0].message.content

            logging.info("Success for create the text!")

            return response_text

        except Exception as e:
            logging.info(e)

