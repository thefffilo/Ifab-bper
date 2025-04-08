from langgraph.graph import StateGraph, END
from nodes.ocr import estrai_testo_da_immagine
from nodes.extract_template import extract_template
from nodes.populate import populate_template
from typing import TypedDict


# Stato globale del grafo
class WorkflowState(TypedDict, total=False):
    testo_ocr: str
    campi_template: str
    template_popolato: str

# Nodo 1: OCR
def ocr_node(state: WorkflowState) -> WorkflowState:
    print("Eseguo OCR...")
    testo = estrai_testo_da_immagine("image.png")
    return {"testo_ocr": testo}

# Nodo 2: Estrazione Template
def template_extractor_node(state: WorkflowState) -> WorkflowState:
    print("Estraggo campi dal testo OCR...")
    testo = state.get("testo_ocr", "")
    if not testo:
        raise ValueError("Nessun testo OCR disponibile.")
    output = extract_template(testo)
    return {"campi_template": output}

# Nodo 3: Popolamento Template
def populate_template_node(state: WorkflowState) -> WorkflowState:
    print("Popolo il template...")
    campi_template = state.get("campi_template", "")
    if not campi_template:
        raise ValueError("Nessun template disponibile.")
    populated_template = populate_template(campi_template)
    return {"template_popolato": populated_template}

# Costruzione del grafo
workflow = StateGraph(WorkflowState)
workflow.add_node("ocr", ocr_node)
workflow.add_node("estrai_template", template_extractor_node)
workflow.add_node("popola_template", populate_template_node) 

workflow.set_entry_point("ocr")
workflow.add_edge("ocr", "estrai_template")
workflow.add_edge("estrai_template", "popola_template") 
workflow.add_edge("popola_template", END)

app = workflow.compile()

# Esecuzione
if __name__ == "__main__":
    print("🚀 Avvio grafo...")
    result = app.invoke({})
    print("\nTemplate popolato:\n", result["template_popolato"])