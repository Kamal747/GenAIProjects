"""
GenAI Chat + Image
--------------------------------------------------
A simple chat app where the assistant can both talk AND generate images.

Flow (per the assignment spec):
    1. Keep messages in st.session_state
    2. On each user input -> call LLM -> get reply
    3. If reply contains "[IMAGE: ...]" -> generate an image from that prompt
    4. Append everything back to messages
    5. Re-render

LLM      : Llama (served via Groq's cloud API — fast, no local install needed)
Image gen: Pollinations.ai free image API (no key needed) by default,
           OR OpenAI's Images API (DALL-E) if OPENAI_API_KEY is set.

Run:
    pip install -r requirements.txt
    streamlit run app.py
    # Then paste your Groq API key into the sidebar (get one free at
    # https://console.groq.com/keys). Alternatively, set it as an env var
    # (GROQ_API_KEY) before launching and it will pre-fill the sidebar field.
"""

import os
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="GenAI Chat + Image", page_icon="🎨", layout="centered")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

IMAGE_TAG_PATTERN = re.compile(r"\[IMAGE:\s*(.*?)\]", re.IGNORECASE | re.DOTALL)

# Stages cycled through while the image is being generated, mimicking ChatGPT's
# "Creating image" progress card.
IMAGE_STAGES = ["Generating image", "Sketching it out", "Rendering details", "Final draft"]


def render_image_card(slot, stage_text: str):
    """Render a ChatGPT-style 'Creating image' placeholder card into the given st.empty() slot."""
    slot.markdown(
        f"""
        <div style="
            position:relative;
            width:100%;
            max-width:520px;
            aspect-ratio:1/1;
            border-radius:16px;
            background-color:#e9e9e9;
            background-image:radial-gradient(circle, #cfcfcf 1.5px, transparent 1.5px);
            background-size:18px 18px;
            display:flex;
            align-items:flex-start;
            justify-content:flex-start;
            padding:20px;
            box-sizing:border-box;
            overflow:hidden;
        ">
            <span style="font-weight:600; font-size:1rem; color:#555;">{stage_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

SYSTEM_PROMPT = """You are a helpful assistant that can both chat normally and request images.

Whenever an image would help answer the user (they ask to see/draw/generate/create/show
a picture of something, or a visual clearly adds value), respond with ONLY a tag of the
exact form:
[IMAGE: <a short, vivid, self-contained description of the image to generate>]
and NOTHING else — no preamble, no "Here you go", no extra commentary. Just the tag.

For everything else (normal conversation, questions, explanations), reply normally with
plain text and do NOT include the [IMAGE: ...] tag."""


# --------------------------------------------------------------------------
# LLM call (Groq chat-completions endpoint — OpenAI-compatible)
# --------------------------------------------------------------------------
def call_llm(messages: list[dict], api_key: str) -> str:
    """Send full conversation history to Groq's API and return the reply text."""
    if not api_key:
        return (
            "⚠️ No Groq API key set. Paste one into the sidebar, or get a free "
            "key at https://console.groq.com/keys"
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        return f"⚠️ Groq API error: {e} — {resp.text[:300]}"
    except Exception as e:
        return f"⚠️ LLM call failed: {e}"


# --------------------------------------------------------------------------
# Image generation
# --------------------------------------------------------------------------
def generate_image(prompt: str, openai_key: str = "") -> str | None:
    """Return an image URL for the given prompt. Uses OpenAI if a key is set, else Pollinations."""
    if openai_key:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/images/generations",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={"model": "dall-e-3", "prompt": prompt, "n": 1, "size": "1024x1024"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["url"]
        except Exception as e:
            st.warning(f"OpenAI image generation failed, falling back: {e}")

    # Free fallback: Pollinations.ai (no key required, returns the image directly)
    encoded_prompt = urllib.parse.quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded_prompt}"


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "image": optional URL}
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.environ.get("GROQ_API_KEY", "")
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get("OPENAI_API_KEY", "")


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
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key (optional)",
        value=st.session_state.openai_api_key,
        type="password",
        placeholder="sk-... (leave blank to use free Pollinations.ai)",
        help="If provided, images are generated with DALL-E 3 instead of the free Pollinations.ai API.",
    )

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------
# Render existing messages
# --------------------------------------------------------------------------
st.title("🎨 GenAI Chat + Image")
st.caption("Chat normally, or ask for a picture — the assistant decides when to generate one.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], use_container_width=True)


# --------------------------------------------------------------------------
# Handle new user input
# --------------------------------------------------------------------------
user_input = st.chat_input("Type a message, or ask me to draw / show / generate something...")

if user_input:
    # 1. Append user message and render it immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Call LLM with full history (role/content only, no image fields needed by the API)
    history_for_llm = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m.get("content")
    ]

    with st.spinner("Thinking..."):
        reply = call_llm(history_for_llm, st.session_state.groq_api_key)

    # 3. Check for an [IMAGE: ...] tag in the reply
    image_prompt = None
    match = IMAGE_TAG_PATTERN.search(reply)
    if match:
        image_prompt = match.group(1).strip()
        # Strip the tag out of the visible text
        reply_text = IMAGE_TAG_PATTERN.sub("", reply).strip()
    else:
        reply_text = reply

    # 4. Render assistant turn — text appears immediately (only for non-image
    #    replies); if an image was requested, show a ChatGPT-style staged
    #    "Creating image" card that cycles through stages while generation
    #    runs in the background, then swaps to the real image.
    image_url = None
    with st.chat_message("assistant"):
        if image_prompt:
            image_slot = st.empty()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(generate_image, image_prompt, st.session_state.openai_api_key)
                stage_idx = 0
                while not future.done():
                    render_image_card(image_slot, IMAGE_STAGES[stage_idx % len(IMAGE_STAGES)])
                    stage_idx += 1
                    time.sleep(1.1)
                image_url = future.result()
            image_slot.image(image_url, use_container_width=True)
        elif reply_text:
            st.markdown(reply_text)

    # 5. Append assistant message (with image info) back into session_state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "" if image_prompt else reply_text,
            "image": image_url,
            "image_prompt": image_prompt,
        }
    )