import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("LLM_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def estrai_testo_da_immagine(percorso_immagine):
    image = Image.open(percorso_immagine)

    model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

    response = model.generate_content(
        [
            "Genera il codice html di questa foto. Cerca di essere più accurato possibile. Non aggiungere testo, riscrivi tutto il testo presente nell'immagine",
            image
        ]
    )

    return response.text
