from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("LLM_API_KEY")

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=GOOGLE_API_KEY)

system_prompt = """
Sei un assistente intelligente incaricato di compilare i campi di un documento/form in modo coerente e realistico.

Ti verrà fornita una **lista di nomi di campo** e una breve **descrizione** per ciascun campo. Il tuo compito è:

1. Restituire tutti i campi presenti nella lista, associando a ciascuno un **valore plausibile**.
2. Restituire l'output come **una lista di coppie nome-campo: valore** separate da due punti, in formato testuale (senza JSON).
3. Non inventare nuovi campi.
4. I valori devono sembrare coerenti tra loro e con il contesto di un modulo formale (es. bancario, legale, finanziario).

### Esempio di input:
[
  {
    "nome": "RICHIEDENTE",
    "descrizione": "Nome e cognome della persona o ente che ha fatto la richiesta."
  },
  {
    "nome": "ISTITUTO DI CREDITO",
    "descrizione": "Banca o società finanziaria a cui è collegata la segnalazione."
  }
]

### Esempio di output atteso:

RICHIEDENTE: Mario Rossi,
ISTITUTO DI CREDITO: UniCredit S.p.A.

Fornisci solo l'output come testo, senza spiegazioni aggiuntive.
"""

def populate_template(query):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    response = chat_model(messages)
    return response.content
