from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage

from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("LLM_API_KEY")

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=GOOGLE_API_KEY)

system_prompt = """
Sei un assistente che estrae la struttura logica di un template da un testo. Il testo fornito rappresenta un modulo o documento compilabile. Il tuo compito è:
1. Identificare i **nomi dei campi** presenti nel testo.
2. Per ciascun campo, fornisci una **descrizione chiara e sintetica** del suo significato o scopo, basandoti sul nome e sul contesto.
3. Restituisci l'output in formato JSON come **array di oggetti**, ognuno con `nome` e `descrizione`.
4. Mantieni **l'ordine dei campi** così come appaiono nel testo.
5. Includi i **campi ripetuti**, senza filtrarli, rispettando sempre l'ordine.
6. Ignora qualsiasi valore compilato o dato inserito nel documento.

### Esempio di output atteso:
Input:
RICHIEDENTE
ISTITUTO DI CREDITO
DATI AGGIORNATI AL

Output:
[
  {
    "nome": "RICHIEDENTE",
    "descrizione": "Nome e cognome della persona o ente che ha fatto la richiesta."
  },
  {
    "nome": "ISTITUTO DI CREDITO",
    "descrizione": "Banca o società finanziaria a cui è collegata la segnalazione."
  },
  {
    "nome": "DATI AGGIORNATI AL",
    "descrizione": "Data dell’ultimo aggiornamento delle informazioni contenute nel documento."
  }
]

Fornisci solo l'output JSON, senza spiegazioni aggiuntive.
"""

def extract_template(query):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    response = chat_model(messages)
    return response.content
