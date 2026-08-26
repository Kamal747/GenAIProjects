# 🏥 Project 8: AI-Powered Multi-Agent Hospital Workflow
### Using LangGraph, Groq, and Streamlit

## 🌐 Live Demo

🔗 https://genaiprojects-mdpfmzy8shbhqdde29i94q.streamlit.app

Models the **complete patient treatment lifecycle** — from admission to
discharge — as a LangGraph state graph. Each medical specialist is an
intelligent agent responsible for one stage of treatment, and all agents
share a single **centralized patient state** (memory) as the patient moves
through the hospital.

## Architecture

```
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
              Follow-up ◆ (conditional decision)
              │                        │
    no further treatment      needs surgery / specialist care
              │                        │
              ▼                        ▼
       Discharge Agent      Cardiothoracic Surgeon Agent
              │                        │
              │                        ▼
              │                 Monitoring Agent
              │                        │
              │                        ▼
              │                 Discharge Agent
              │                        │
              └───────────┬────────────┘
                           ▼
                          End
```

## Agents (Nodes)

| Agent | Role |
|--------|------|
| 🧾 **Patient Registration Agent** | Records patient intake details and presenting symptoms |
| 🩺 **General Physician Agent** | Initial assessment, refers onward to Cardiology |
| ❤️ **Cardiologist Agent** | Produces the cardiology report AND the **Follow-up decision** — the conditional edge that determines whether the patient needs surgery or can be discharged |
| 🔪 **Cardiothoracic Surgeon Agent** | Recommends and documents the procedure (only reached if surgery is required) |
| 📈 **Monitoring Agent** | Post-operative monitoring, confirms stability before discharge |
| 🏠 **Discharge Agent** | Produces the final discharge summary — reached either directly from Cardiology or after Surgeon → Monitor |

## LangGraph concepts demonstrated

- **Nodes** — one per specialist agent (`patient_registration`, `general_physician`, `cardiologist`, `surgeon`, `monitor`, `discharge`)
- **Edges** — fixed transitions: Registration → GP → Cardiologist, and Surgeon → Monitor → Discharge
- **Conditional Edges** — the "Follow-up" decision after Cardiologist dynamically routes to either `surgeon` or `discharge` based on the LLM's assessment
- **Shared State** — a single `PatientState` TypedDict threaded through every node, so each agent has full context of everything that happened before it
- **Start / End** — `graph.set_entry_point("patient_registration")` and `END`

## Tech stack

| Component        | Choice                                                       |
|-------------------|-----------------------------------------------------------------|
| Orchestration     | [LangGraph](https://langchain-ai.github.io/langgraph/) (`StateGraph`, conditional routing, shared state) |
| LLM               | Llama 3.1 8B Instant via [Groq API](https://console.groq.com)   |
| UI                | Streamlit                                                        |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Sign up at [console.groq.com/keys](https://console.groq.com/keys).

### 3. Run the app
```bash
streamlit run project8.py
```

### 4. Run a patient through the workflow
Paste your Groq key into the sidebar, fill in the patient intake form
(name, age, symptoms), and click **Run patient through hospital workflow**.

Try these two symptom examples to see both branches of the conditional edge:
- *"Severe chest pain, diagnosed with a major arterial blockage requiring bypass surgery"* → routes through **Surgeon → Monitor → Discharge**
- *"Occasional mild palpitations, stable blood pressure, no chest pain"* → routes **directly to Discharge**

The **Path taken** banner and the expandable sections show exactly which
agents were visited and what each one recorded in the shared patient state.

## Environment variables

| Variable         | Default                   | Purpose                                  |
|-------------------|-----------------------------|--------------------------------------------|
| `GROQ_API_KEY`    | *(unset)*                  | Pre-fills the sidebar API key field         |
| `GROQ_MODEL`      | `llama-3.1-8b-instant`     | Which Groq-hosted model powers every agent  |

## Notes

- **This is a demo, not a real clinical system.** All specialist reasoning is
  simulated by an LLM — no real patient records, diagnostic equipment, or
  hospital systems are involved.
- The Cardiologist agent's decision (`SURGERY_REQUIRED` vs `DISCHARGE`) is
  parsed from a structured marker in its own LLM output — this is a simple,
  inspectable way to let an LLM drive a LangGraph conditional edge.
- **Extending this project** (per the original brief) could include:
  - Additional specialists (e.g. Anesthesiologist, Radiologist) as new nodes
  - RAG-based retrieval of the patient's prior medical history
  - A human-in-the-loop approval step before surgery is confirmed
  - Real-time monitoring with periodic re-evaluation (a loop back to Cardiologist)
  - Multi-patient support with a patient queue instead of a single intake form

<!-- Append this section to Project8_MultiAgent_Hospital/README.md, after "## Notes" -->

## 🧠 ReAct-style Reasoning Logging (additive)

Every agent node now generates output in explicit **Thought → Action →
Observation** form before returning its note/report:

- **Thought** — the agent's stated reasoning before acting
- **Action** — the short decision it's taking
- **Observation** — the actual clinical note/report (what previously was
  the entire LLM response) — this is what still flows into `PatientState`,
  so the LangGraph nodes/edges/routing are completely unchanged

Each step is parsed and logged in `react_logger.py`:
- Printed to console in a structured block per agent call
- Appended as one JSON line to `react_logs.jsonl` (path overridable via
  `REACT_LOG_FILE` env var)

If an LLM ever skips the format, the parser falls back to using the whole
response as the Observation — so the workflow can never break because of
a malformed ReAct response.

**Files added:** `react_logger.py`
**Files changed:** `project8.py` (prompts + logging calls only — no
node/edge/state structure changed)

## 🔀 Alternate Framework Version: CrewAI (`crewai_version/`)

A standalone, scoped rebuild of the same hospital workflow using
[CrewAI](https://docs.crewai.com/) instead of LangGraph — 3 of the 6
agents (Patient Registration → General Physician → Cardiologist),
same Groq `llama-3.1-8b-instant` model, same shared-patient-state idea
(passed through CrewAI's task `context=[...]`).

This lives entirely in `crewai_version/` and does not touch or import
`project8.py` — the original LangGraph app is untouched.

```bash
cd crewai_version
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...
python crewai_hospital.py
```

**Files added:** `crewai_version/crewai_hospital.py`,
`crewai_version/requirements.txt`
