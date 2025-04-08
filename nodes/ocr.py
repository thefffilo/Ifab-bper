from google.cloud import vision
import io
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'sodium-carving-456119-f4-98ef0322e0f8.json'

def estrai_testo_da_immagine(percorso_immagine):
    client = vision.ImageAnnotatorClient()

    with io.open(percorso_immagine, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("Nessun testo rilevato.")
        return ""

    testo_completo = texts[0].description
    # print("Testo rilevato:\n", testo_completo)

    return testo_completo