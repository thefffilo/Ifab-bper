# nodes/evaluator.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("LLM_API_KEY")

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=GOOGLE_API_KEY)

system_prompt = """
Sei un assistente esperto nella valutazione di generazione dati.

Riceverai:
1. Il testo originale in formato html di un report.
2. Un report che dovrebbe avere la stessa struttura ma con dati inventati, lo scopo è cercare di generare un report più simile possibile con dati sintetici.
3. Una lista di parametri di valutazione (ogni parametro ha nome e descrizione).

Per ciascun parametro:
- Analizza quanto il template rispetta il criterio.
- Dai un punteggio da 1 a 10.

Restituisci la valutazione come elenco puntato, uno per ogni parametro.

### Formato atteso:
- completezza (8/10): La maggior parte dei dati è presente, ma mancano alcune informazioni minori.
- coerenza (9/10): I valori sono coerenti con il testo OCR.
"""

def valuta_output_template(testo_ocr: str, template: str, parametri: list) -> str:
    if not testo_ocr or not template or not parametri:
        return "Errore: Input incompleto per la valutazione."

    # Costruisci rappresentazione testuale dei parametri
    parametri_text = "\n".join(
        [f"- {p['nome']}: {p['descrizione']}" for p in parametri]
    )

    query = f"""### Testo OCR:
{testo_ocr}

### Template Popolato:
{template}

### Parametri di Valutazione:
{parametri_text}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    response = chat_model(messages)
    return response.content
