"""
Project 8: AI-Powered Multi-Agent Hospital Workflow
Using LangGraph, Groq, and Streamlit
--------------------------------------------------------------------------
Models the complete patient treatment lifecycle from admission to discharge
as a LangGraph state graph, where each medical specialist is an intelligent
agent responsible for one stage of treatment, sharing a single patient state
(centralized memory) across the whole journey.

Workflow (matches the "Nodes / Edges / State" architecture diagram):

    Start
      │
      ▼
    Patient Registration Agent
      │
      ▼
    General Physician Agent
      │
      ▼
    Cardiologist Agent
      │
      ▼
    Follow-up  ◆ (conditional decision — dynamic referral)
      │                              │
      │ no further treatment          │ needs surgery / specialist care
      ▼                              ▼
    Discharge Agent            Cardiothoracic Surgeon Agent
      │                              │
      │                              ▼
      │                        Monitoring Agent
      │                              │
      │                              ▼
      │                        Discharge Agent
      │                              │
      └──────────────┬───────────────┘
                      ▼
                     End

Shared State (patient memory) persists across every node: patient info,
diagnostic reports, prescriptions/treatment notes, and the decision that
drove each conditional referral — so every agent has full context of what
came before it.

LangGraph concepts demonstrated:
    - Nodes            : one per specialist agent
    - Edges             : fixed transitions (Registration -> GP -> Cardiologist)
    - Conditional Edges : dynamic referral after Cardiologist (Follow-up)
    - Shared State      : PatientState TypedDict threaded through every node
    - Start / End       : graph.set_entry_point(...) / END

Tech stack:
    Orchestration : LangGraph (StateGraph, conditional routing, shared state)
    LLM           : Groq (Llama 3.1) — simulates each specialist's reasoning
    UI            : Streamlit

Run:
    pip install -r requirements.txt
    streamlit run project8.py
    # Paste your Groq API key into the sidebar (free key: https://console.groq.com/keys)
"""

import os
from typing import TypedDict, Literal

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Hospital Multi-Agent Workflow", page_icon="🏥", layout="wide")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

STAGE_LABELS = {
    "patient_registration": "🧾 Patient Registration Agent",
    "general_physician": "🩺 General Physician Agent",
    "cardiologist": "❤️ Cardiologist Agent",
    "surgeon": "🔪 Cardiothoracic Surgeon Agent",
    "monitor": "📈 Monitoring Agent",
    "discharge": "🏠 Discharge Agent",
}


# --------------------------------------------------------------------------
# Shared State (centralized patient memory across every node)
# --------------------------------------------------------------------------
class PatientState(TypedDict):
    patient_name: str
    age: str
    symptoms: str
    registration_notes: str
    gp_notes: str
    cardiology_report: str
    needs_surgery: bool
    surgeon_notes: str
    monitoring_notes: str
    discharge_summary: str
    stage_log: list[str]  # ordered list of stage keys visited, for the UI timeline


def _log_stage(state: PatientState, stage: str) -> list[str]:
    return state["stage_log"] + [stage]


# --------------------------------------------------------------------------
# Node: Patient Registration Agent
# --------------------------------------------------------------------------
def patient_registration_node(state: PatientState, llm: ChatGroq) -> PatientState:
    system = SystemMessage(content="""You are a hospital Patient Registration agent. Write a brief,
professional intake record summarizing the patient's basic details and presenting complaint, ready
to be handed off to the General Physician. Keep it to 2-3 sentences.""")
    human = HumanMessage(
        content=f"Patient name: {state['patient_name']}\nAge: {state['age']}\nPresenting symptoms: {state['symptoms']}"
    )
    notes = llm.invoke([system, human]).content.strip()
    return {**state, "registration_notes": notes, "stage_log": _log_stage(state, "patient_registration")}


# --------------------------------------------------------------------------
# Node: General Physician Agent
# --------------------------------------------------------------------------
def general_physician_node(state: PatientState, llm: ChatGroq) -> PatientState:
    system = SystemMessage(content="""You are a hospital General Physician agent. Review the patient's
registration record and symptoms, provide a brief initial assessment, and refer the patient onward to
Cardiology for further evaluation (this hospital's workflow always routes through Cardiology next).
Keep it to 3-4 sentences.""")
    human = HumanMessage(
        content=f"Registration notes: {state['registration_notes']}\nSymptoms: {state['symptoms']}"
    )
    notes = llm.invoke([system, human]).content.strip()
    return {**state, "gp_notes": notes, "stage_log": _log_stage(state, "general_physician")}


# --------------------------------------------------------------------------
# Node: Cardiologist Agent  (produces the Follow-up decision)
# --------------------------------------------------------------------------
def cardiologist_node(state: PatientState, llm: ChatGroq) -> PatientState:
    system = SystemMessage(content="""You are a hospital Cardiologist agent. Review the GP's referral
notes and the patient's symptoms, and produce a short cardiology report (3-5 sentences).

Then make a referral decision: if the case requires cardiothoracic surgery or further specialist
intervention, end your report with a new line exactly:
DECISION: SURGERY_REQUIRED
Otherwise, if the patient can be safely discharged with medication/advice, end with exactly:
DECISION: DISCHARGE

Base this decision reasonably on the symptoms described (e.g. severe blockage, structural heart
issues, or explicit mentions of needing surgery/procedure should lead to SURGERY_REQUIRED; mild,
manageable, or stable conditions should lead to DISCHARGE).""")
    human = HumanMessage(
        content=f"GP notes: {state['gp_notes']}\nSymptoms: {state['symptoms']}"
    )
    result = llm.invoke([system, human]).content.strip()

    needs_surgery = "SURGERY_REQUIRED" in result.upper()
    report = result.split("DECISION:")[0].strip()

    return {
        **state,
        "cardiology_report": report,
        "needs_surgery": needs_surgery,
        "stage_log": _log_stage(state, "cardiologist"),
    }


# --------------------------------------------------------------------------
# Conditional routing: the "Follow-up" diamond in the diagram
# --------------------------------------------------------------------------
def route_after_cardiologist(state: PatientState) -> Literal["surgeon", "discharge"]:
    return "surgeon" if state["needs_surgery"] else "discharge"


# --------------------------------------------------------------------------
# Node: Cardiothoracic Surgeon Agent
# --------------------------------------------------------------------------
def surgeon_node(state: PatientState, llm: ChatGroq) -> PatientState:
    system = SystemMessage(content="""You are a hospital Cardiothoracic Surgeon agent. Based on the
cardiology report, describe the recommended procedure and immediate post-procedure notes in 3-4
sentences. Hand off to the Monitoring team next.""")
    human = HumanMessage(content=f"Cardiology report: {state['cardiology_report']}")
    notes = llm.invoke([system, human]).content.strip()
    return {**state, "surgeon_notes": notes, "stage_log": _log_stage(state, "surgeon")}


# --------------------------------------------------------------------------
# Node: Monitoring Agent
# --------------------------------------------------------------------------
def monitor_node(state: PatientState, llm: ChatGroq) -> PatientState:
    system = SystemMessage(content="""You are a hospital Monitoring agent. Based on the surgeon's notes,
write a brief post-operative monitoring summary (vitals stability, recovery progress) confirming the
patient is stable and ready for discharge. 2-3 sentences.""")
    human = HumanMessage(content=f"Surgeon notes: {state['surgeon_notes']}")
    notes = llm.invoke([system, human]).content.strip()
    return {**state, "monitoring_notes": notes, "stage_log": _log_stage(state, "monitor")}


# --------------------------------------------------------------------------
# Node: Discharge Agent (reached either directly from Cardiologist or via Surgeon -> Monitor)
# --------------------------------------------------------------------------
def discharge_node(state: PatientState, llm: ChatGroq) -> PatientState:
    context_parts = [
        f"Registration notes: {state['registration_notes']}",
        f"GP notes: {state['gp_notes']}",
        f"Cardiology report: {state['cardiology_report']}",
    ]
    if state["needs_surgery"]:
        context_parts.append(f"Surgeon notes: {state['surgeon_notes']}")
        context_parts.append(f"Monitoring notes: {state['monitoring_notes']}")

    system = SystemMessage(content="""You are a hospital Discharge agent. Using the full patient
journey provided, write a concise discharge summary: diagnosis, treatment received, medications/advice,
and any follow-up recommendations. 4-6 sentences, patient-friendly tone.""")
    human = HumanMessage(content="\n".join(context_parts))
    summary = llm.invoke([system, human]).content.strip()
    return {**state, "discharge_summary": summary, "stage_log": _log_stage(state, "discharge")}


# --------------------------------------------------------------------------
# Build the LangGraph workflow
# --------------------------------------------------------------------------
def build_graph(llm: ChatGroq):
    graph = StateGraph(PatientState)

    graph.add_node("patient_registration", lambda s: patient_registration_node(s, llm))
    graph.add_node("general_physician", lambda s: general_physician_node(s, llm))
    graph.add_node("cardiologist", lambda s: cardiologist_node(s, llm))
    graph.add_node("surgeon", lambda s: surgeon_node(s, llm))
    graph.add_node("monitor", lambda s: monitor_node(s, llm))
    graph.add_node("discharge", lambda s: discharge_node(s, llm))

    graph.set_entry_point("patient_registration")
    graph.add_edge("patient_registration", "general_physician")
    graph.add_edge("general_physician", "cardiologist")

    # Conditional "Follow-up" decision after Cardiologist
    graph.add_conditional_edges(
        "cardiologist",
        route_after_cardiologist,
        {"surgeon": "surgeon", "discharge": "discharge"},
    )

    graph.add_edge("surgeon", "monitor")
    graph.add_edge("monitor", "discharge")
    graph.add_edge("discharge", END)

    return graph.compile()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.environ.get("GROQ_API_KEY", "")
if "last_run" not in st.session_state:
    st.session_state.last_run = None  # holds the final PatientState after a workflow run


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.session_state.groq_api_key = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_...",
        help="Get a free key at https://console.groq.com/keys",
    )

    st.caption(f"Model: `{GROQ_MODEL}` (set env var `GROQ_MODEL` to change)")

    st.divider()

    st.subheader("🧠 Agents in this workflow")
    for label in STAGE_LABELS.values():
        st.markdown(f"- {label}")

    st.divider()

    # ---------------- Clear Chat ----------------
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []

        # Optional: Clear workflow state if present
        if "patient_state" in st.session_state:
            del st.session_state.patient_state

        if "workflow_state" in st.session_state:
            del st.session_state.workflow_state

        st.rerun()

    st.divider()

    st.caption(
        "⚠️ This is a demo simulating specialist reasoning with an LLM. "
        "It does not provide real medical advice or access real hospital systems."
    )


# --------------------------------------------------------------------------
# Main UI: patient intake + workflow execution
# --------------------------------------------------------------------------
st.title("🏥 AI-Powered Multi-Agent Hospital Workflow")
st.caption(
    "Patient Registration → General Physician → Cardiologist → Follow-up (conditional) → "
    "Surgeon/Monitor → Discharge — a LangGraph workflow with shared patient state across every agent."
)

with st.form("intake_form"):
    st.subheader("🧾 New Patient Intake")
    col1, col2 = st.columns([2, 1])
    with col1:
        patient_name = st.text_input("Patient name", placeholder="e.g. Arun Kumar")
    with col2:
        age = st.text_input("Age", placeholder="e.g. 54")
    symptoms = st.text_area(
        "Presenting symptoms",
        placeholder="e.g. Chest tightness on exertion, shortness of breath for the past 2 weeks...",
        height=100,
    )
    submitted = st.form_submit_button("▶️ Run patient through hospital workflow", type="primary")

if submitted:
    if not st.session_state.groq_api_key:
        st.error("⚠️ Please paste your Groq API key into the sidebar first.")
    elif not patient_name or not symptoms:
        st.error("⚠️ Please provide at least a patient name and symptoms.")
    else:
        try:
            llm = ChatGroq(model=GROQ_MODEL, groq_api_key=st.session_state.groq_api_key, temperature=0.3)
            graph = build_graph(llm)

            initial_state: PatientState = {
                "patient_name": patient_name,
                "age": age or "unknown",
                "symptoms": symptoms,
                "registration_notes": "",
                "gp_notes": "",
                "cardiology_report": "",
                "needs_surgery": False,
                "surgeon_notes": "",
                "monitoring_notes": "",
                "discharge_summary": "",
                "stage_log": [],
            }

            with st.spinner("Running patient through the hospital workflow..."):
                final_state = graph.invoke(initial_state)

            st.session_state.last_run = final_state
        except Exception as e:
            st.error(f"⚠️ Workflow failed: {e}")


# --------------------------------------------------------------------------
# Display the patient's journey (shared state timeline)
# --------------------------------------------------------------------------
if st.session_state.last_run:
    state = st.session_state.last_run

    st.divider()
    st.subheader(f"📋 Patient Journey — {state['patient_name']} (Age: {state['age']})")

    path_taken = " → ".join(STAGE_LABELS[s] for s in state["stage_log"])
    st.info(f"**Path taken:** {path_taken}")

    with st.expander(STAGE_LABELS["patient_registration"], expanded=True):
        st.markdown(state["registration_notes"])

    with st.expander(STAGE_LABELS["general_physician"]):
        st.markdown(state["gp_notes"])

    with st.expander(STAGE_LABELS["cardiologist"]):
        st.markdown(state["cardiology_report"])
        decision = "🔪 Surgery required → referred to Surgeon" if state["needs_surgery"] else "🏠 No surgery needed → referred directly to Discharge"
        st.markdown(f"**Follow-up decision:** {decision}")

    if state["needs_surgery"]:
        with st.expander(STAGE_LABELS["surgeon"]):
            st.markdown(state["surgeon_notes"])
        with st.expander(STAGE_LABELS["monitor"]):
            st.markdown(state["monitoring_notes"])

    with st.expander(STAGE_LABELS["discharge"], expanded=True):
        st.markdown(state["discharge_summary"])