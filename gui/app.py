"""Streamlit prototype for Bash Guardian AI.

This lightweight GUI is intended for portfolio/demo use. It previews the
project's persona dataset, prompt/guardrail artifacts, and local evaluation
workflow without requiring a fully fine-tuned model checkpoint.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "bash_persona_sample.jsonl"
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system_prompt.md"
GUARDRAILS_PATH = ROOT / "prompts" / "guardrails.md"
EVALUATOR_PATH = ROOT / "tests" / "evaluate_persona.py"


def load_jsonl(path: Path) -> list[dict]:
    examples: list[dict] = []
    if not path.exists():
        return examples

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def read_text(path: Path) -> str:
    if not path.exists():
        return f"Missing file: {path}"
    return path.read_text(encoding="utf-8")


def simple_bash_response(user_message: str, examples: list[dict]) -> str:
    """Return a deterministic prototype response.

    This is not model inference. It is a small demo responder that shows the
    intended supportive tone while the LoRA workflow is being prepared.
    """
    lowered = user_message.lower()

    if any(word in lowered for word in ["overwhelmed", "too much", "start", "avoid"]):
        return (
            "Take one breath first. We do not need to solve the whole mountain. "
            "Pick one tiny step that takes less than five minutes, and start there."
        )

    if any(word in lowered for word in ["sad", "scared", "anxious", "panic"]):
        return (
            "I am here with you. Name one thing you can see, press your feet into "
            "the floor, and let this moment get smaller and safer before we choose "
            "the next step."
        )

    if examples:
        return examples[0].get("output", "Start with one small, concrete step.")

    return "Start with one small, concrete step. Keep it gentle and concrete."


def run_evaluator() -> tuple[int, str]:
    if not EVALUATOR_PATH.exists():
        return 1, "Evaluator script not found."

    completed = subprocess.run(
        [sys.executable, str(EVALUATOR_PATH)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout
    if completed.stderr:
        output += "\n\nSTDERR:\n" + completed.stderr
    return completed.returncode, output


st.set_page_config(page_title="Bash Guardian AI", page_icon="🐺", layout="wide")

st.title("🐺 Bash Guardian AI")
st.caption("Prototype interface for testing persona behavior, prompt structure, and guardrail evaluation.")

examples = load_jsonl(DATASET_PATH)

with st.sidebar:
    st.header("Project Status")
    st.write("**Base model target:** `meta-llama/Llama-3.2-3B-Instruct`")
    st.write("**Fine-tuning method:** LoRA / PEFT scaffold")
    st.write("**Dataset examples loaded:**", len(examples))
    st.info("This demo uses a deterministic prototype responder, not a completed fine-tuned model checkpoint.")

chat_tab, evaluator_tab, docs_tab, data_tab = st.tabs(
    ["Chat Prototype", "Evaluator", "Prompt & Guardrails", "Dataset Preview"]
)

with chat_tab:
    st.subheader("Chat Prototype")
    user_message = st.text_area("Message", placeholder="I feel overwhelmed and do not know where to start.")

    if st.button("Generate prototype response"):
        if not user_message.strip():
            st.warning("Type a message first.")
        else:
            st.markdown("**Bash Guardian AI:**")
            st.write(simple_bash_response(user_message, examples))
            st.caption("Prototype response generated from rule-based demo logic for portfolio testing.")

with evaluator_tab:
    st.subheader("Persona / Guardrail Evaluation")
    st.write("Runs the local evaluator against the sample companion dataset.")

    if st.button("Run evaluation"):
        exit_code, output = run_evaluator()
        if exit_code == 0:
            st.success("Evaluation completed.")
        else:
            st.error("Evaluation failed or returned a non-zero exit code.")
        st.code(output, language="text")

with docs_tab:
    st.subheader("System Prompt")
    st.markdown(read_text(SYSTEM_PROMPT_PATH))

    st.subheader("Guardrails")
    st.markdown(read_text(GUARDRAILS_PATH))

with data_tab:
    st.subheader("Dataset Preview")
    if not examples:
        st.warning("No dataset examples found.")
    else:
        for index, example in enumerate(examples, start=1):
            with st.expander(f"Example {index}"):
                st.markdown("**Input**")
                st.write(example.get("input", ""))
                st.markdown("**Output**")
                st.write(example.get("output", ""))
