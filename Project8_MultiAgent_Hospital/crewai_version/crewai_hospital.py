"""
CrewAI Alternate Implementation — Multi-Agent Hospital Workflow (scoped)
--------------------------------------------------------------------------
Same hospital-workflow concept as ../project8.py (the LangGraph version),
rebuilt in CrewAI to demonstrate the same multi-agent design ported to a
different orchestration framework.

Scope: 3 of the original 6 agents — Patient Registration, General
Physician, Cardiologist — kept intentionally small since this is an
alternate implementation, not a replacement of the full LangGraph workflow.

This file is completely standalone: it does not import, modify, or depend
on project8.py in any way. The original LangGraph app is untouched.

Tech stack:
    Orchestration : CrewAI (Agent, Task, Crew, Process.sequential)
    LLM           : Groq (Llama 3.1 8B Instant) via CrewAI's built-in LLM class
    Shared state  : a single PatientState dict passed forward through task
                    context — the same shared-memory idea as project8.py's
                    PatientState TypedDict, just CrewAI's task-context form
                    of it instead of LangGraph's graph state.

Run:
    pip install -r requirements.txt
    export GROQ_API_KEY=gsk_...          # https://console.groq.com/keys
    python crewai_hospital.py
"""

import json
import os
from datetime import datetime, timezone
from typing import TypedDict

from crewai import Agent, Task, Crew, Process, LLM

# --------------------------------------------------------------------------
# Workaround for a known CrewAI/LiteLLM bug (crewAI issue #5886): CrewAI
# injects an Anthropic-style "cache_breakpoint" field into system messages
# for ALL providers, but Groq's API rejects that field. This monkeypatch
# makes the cache-breakpoint marker a no-op so Groq calls go through clean.
# Safe to keep even after upgrading crewai, since it just disables an
# optional caching hint that Groq doesn't support anyway.
try:
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg
except (ImportError, AttributeError):
    pass

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
LOG_FILE = os.environ.get("CREWAI_LOG_FILE", "crewai_logs.jsonl")


def log_patient_run(state: "PatientState"):
    """
    Append this patient's journey (same 3-agent outputs as printed to
    console) as one JSON line to LOG_FILE — mirrors project8.py's
    react_logs.jsonl so both versions leave a file-based trace, not just
    console output.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patient_name": state["patient_name"],
        "age": state["age"],
        "symptoms": state["symptoms"],
        "registration_notes": state["registration_notes"],
        "gp_notes": state["gp_notes"],
        "cardiology_report": state["cardiology_report"],
        "decision": "SURGERY_REQUIRED" if state["needs_surgery"] else "DISCHARGE",
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[crewai_hospital] Warning: could not write to {LOG_FILE}: {e}")


class PatientState(TypedDict):
    patient_name: str
    age: str
    symptoms: str
    registration_notes: str
    gp_notes: str
    cardiology_report: str
    needs_surgery: bool


def build_llm() -> LLM:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GROQ_API_KEY before running (free key: https://console.groq.com/keys)"
        )
    # CrewAI routes non-OpenAI models through LiteLLM's "provider/model" naming.
    return LLM(model=f"groq/{GROQ_MODEL}", api_key=api_key, temperature=0.3)


def build_agents(llm: LLM):
    registration_agent = Agent(
        role="Patient Registration Specialist",
        goal="Produce a concise, professional intake record for the patient.",
        backstory=(
            "You work the hospital front desk. You turn raw patient details into a clean "
            "2-3 sentence intake record ready for the General Physician."
        ),
        llm=llm,
        verbose=True,
    )

    gp_agent = Agent(
        role="General Physician",
        goal="Give an initial assessment and refer the patient to Cardiology.",
        backstory=(
            "You are the first doctor to see the patient. You review the intake record and "
            "symptoms, give a brief assessment, and always refer onward to Cardiology "
            "(this hospital's workflow always routes through Cardiology next)."
        ),
        llm=llm,
        verbose=True,
    )

    cardiologist_agent = Agent(
        role="Cardiologist",
        goal="Produce a cardiology report and a clear SURGERY_REQUIRED / DISCHARGE decision.",
        backstory=(
            "You are a senior cardiologist. You review the GP's referral and the patient's "
            "symptoms, write a 3-5 sentence cardiology report, and end it with a decision marker."
        ),
        llm=llm,
        verbose=True,
    )

    return registration_agent, gp_agent, cardiologist_agent


def build_tasks(agents, state: PatientState):
    registration_agent, gp_agent, cardiologist_agent = agents

    registration_task = Task(
        description=(
            f"Patient name: {state['patient_name']}\n"
            f"Age: {state['age']}\n"
            f"Presenting symptoms: {state['symptoms']}\n\n"
            "Write a brief, professional intake record (2-3 sentences)."
        ),
        expected_output="A 2-3 sentence intake record.",
        agent=registration_agent,
    )

    gp_task = Task(
        description=(
            "Using the intake record above and the patient's symptoms, provide a brief initial "
            "assessment (3-4 sentences) and note that the patient is being referred to Cardiology."
        ),
        expected_output="A 3-4 sentence GP assessment ending in a referral to Cardiology.",
        agent=gp_agent,
        context=[registration_task],
    )

    cardiology_task = Task(
        description=(
            "Using the GP's referral notes and the patient's symptoms, write a 3-5 sentence "
            "cardiology report. Then end your output with exactly one new line:\n"
            "DECISION: SURGERY_REQUIRED\n"
            "or\n"
            "DECISION: DISCHARGE\n"
            "Base this on the symptoms described (severe blockage / structural issues / explicit "
            "surgery mentions -> SURGERY_REQUIRED; mild/stable -> DISCHARGE)."
        ),
        expected_output="A cardiology report followed by a DECISION: line.",
        agent=cardiologist_agent,
        context=[gp_task],
    )

    return registration_task, gp_task, cardiology_task


def run_patient(state: PatientState) -> PatientState:
    llm = build_llm()
    agents = build_agents(llm)
    tasks = build_tasks(agents, state)

    crew = Crew(
        agents=list(agents),
        tasks=list(tasks),
        process=Process.sequential,
        verbose=True,
    )
    crew.kickoff()

    registration_task, gp_task, cardiology_task = tasks
    state["registration_notes"] = str(registration_task.output)
    state["gp_notes"] = str(gp_task.output)

    cardiology_raw = str(cardiology_task.output)
    state["needs_surgery"] = "SURGERY_REQUIRED" in cardiology_raw.upper()
    state["cardiology_report"] = cardiology_raw.split("DECISION:")[0].strip()

    return state


if __name__ == "__main__":
    initial_state: PatientState = {
        "patient_name": "Arun Kumar",
        "age": "54",
        "symptoms": "Chest tightness on exertion, shortness of breath for the past 2 weeks",
        "registration_notes": "",
        "gp_notes": "",
        "cardiology_report": "",
        "needs_surgery": False,
    }

    final_state = run_patient(initial_state)
    log_patient_run(final_state)

    print("\n" + "=" * 70)
    print(f"Patient Journey — {final_state['patient_name']} (Age: {final_state['age']})")
    print("=" * 70)
    print(f"\n[Registration]\n{final_state['registration_notes']}")
    print(f"\n[General Physician]\n{final_state['gp_notes']}")
    print(f"\n[Cardiologist]\n{final_state['cardiology_report']}")
    print(f"\nDecision: {'SURGERY_REQUIRED' if final_state['needs_surgery'] else 'DISCHARGE'}")
