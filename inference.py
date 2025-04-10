import json
from langgraph.graph import StateGraph, END
from nodes.analyze_image import estrai_testo_da_immagine
from nodes.extract_template import extract_template
from nodes.populate import populate_template
from nodes.evaluator import valuta_output_template
from nodes.generate_report import create_report
from typing import TypedDict, Any


# Stato globale del grafo
class WorkflowState(TypedDict, total=False):
    testo_ocr: str
    campi_template: str
    template_popolato: str
    report_finale: str
    valutazione: Any
    parametri_valutazione: list

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

# Nodo 4: Genera report finale
def generate_report_node(state: WorkflowState) -> WorkflowState:
    print("Genero il report...")
    template_popolato = state.get("template_popolato", "")
    if not template_popolato:
        raise ValueError("Nessun contenuto disponibile.")
    report_finale = create_report(template_popolato)
    return {"report_finale": report_finale}

# Nodo 5: Valutazione
def evaluation_node(state: WorkflowState) -> WorkflowState:
    print("Valuto il risultato...")
    testo_ocr = state.get("testo_ocr", "")
    template = state.get("report_finale", "")
    parametri = state.get("parametri_valutazione", [])
    valutazione = valuta_output_template(testo_ocr, template, parametri)
    return {"valutazione": valutazione}


# Costruzione del grafo
workflow = StateGraph(WorkflowState)
workflow.add_node("ocr", ocr_node)
workflow.add_node("estrai_template", template_extractor_node)
workflow.add_node("popola_template", populate_template_node)
workflow.add_node("genera_report", generate_report_node)
workflow.add_node("valuta_output", evaluation_node)

workflow.set_entry_point("ocr")
workflow.add_edge("ocr", "estrai_template")
workflow.add_edge("estrai_template", "popola_template")
workflow.add_edge("popola_template", "genera_report")
workflow.add_edge("genera_report", "valuta_output")
workflow.add_edge("valuta_output", END)

app = workflow.compile()

# Esecuzione
if __name__ == "__main__":
    print("🚀 Avvio grafo...")

    # Caricamento parametri da file JSON
    with open("parametri_valutazione.json", "r", encoding="utf-8") as f:
        parametri = json.load(f)

    result = app.invoke({
        "parametri_valutazione": parametri
    })

    print("\nReport finale:\n", result["report_finale"])
    print("\nValutazione:\n", result["valutazione"])
