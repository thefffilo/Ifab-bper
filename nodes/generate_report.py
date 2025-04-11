from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import pdfkit

from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("LLM_API_KEY")

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", google_api_key=GOOGLE_API_KEY)

system_prompt = """
Sei un assistente intelligente incaricato di creare un report dato un esempio e il contenuto.

Riceverai in input il codice html di un report e il testo di un nuovo report. 
Devi sostituire le informazioni nel codice html con il nuovo contenuto senza modificare la struttura del codice.

Fornisci in output solo il codice html del nuovo report, non aggiungere nessun'altra stringa.

CODICE HTML:
{html}
"""

def create_report(query, html_code):
    sys_prompt = system_prompt.format(html=html_code)
    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=query)
    ]
    response = chat_model(messages)
    pdfkit.from_string(response.content, 'report.pdf')
    return response.content
