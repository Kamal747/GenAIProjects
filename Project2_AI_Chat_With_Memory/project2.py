import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

st.set_page_config(
    page_title="Multi-User Conversational AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multi-User Conversational AI")
st.caption("Built with LangChain • Groq • Llama 3.3 • Session Memory")
st.divider()

# ✅ FIX 1: Persistent memory store
if "store" not in st.session_state:
    st.session_state.store = {}

# --- Session state for UI messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Conversation Statistics
if "stats" not in st.session_state:
    st.session_state.stats = {
        "words": 0,
        "responses": 0,
        "response_time": 0
    }

# --- Display chat history (UI) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Professional Sidebar ---
with st.sidebar:

    st.title("🤖 AI Chat Assistant")

    st.markdown("---")

    st.subheader("🔑 Authentication")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API Key"
    )

    st.markdown("---")

    st.subheader("🧠 Model Settings")

    model = st.selectbox(
        "Choose Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        2.0,
        0.3,
        0.1
    )

    st.markdown("---")
    st.subheader("📊 Conversation Statistics")

    st.metric(
    "Messages",
    len(st.session_state.get("messages", []))
    )

    st.metric(
        "AI Responses",
        st.session_state.stats["responses"]
    )

    st.metric(
        "Words Generated",
        st.session_state.stats["words"]
    )

    st.metric(
        "Response Time",
        f'{st.session_state.stats["response_time"]} sec'
    )

    st.metric(
        "Current Model",
        model
    )

    st.subheader("💾 Session")

    session_id = st.text_input(
        "Session ID",
        value="user1"
    )

    st.markdown("---")

# --- User input ---
user_input = st.chat_input("Type your message...")

# --- Model setup ---
if api_key:
    llm = ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=temperature
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "{message}")
    ])

    chain = prompt_template | llm

    # ✅ FIX 2: Use session_state store
    def get_history(session_id: str):
        store = st.session_state.store
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chat_with_memory = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_history,
        input_messages_key="message",
        history_messages_key="chat_history"
    )

# --- When user sends message ---
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    if not api_key:
        st.error("Enter API key in sidebar")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()

                response = chat_with_memory.invoke(
                    {"message": user_input},
                    {"configurable": {"session_id": session_id}}
                )

                end_time = time.time()

                response_time = round(end_time - start_time, 2)

                reply = response.content
                st.session_state.stats["words"] += len(reply.split())

                st.session_state.stats["responses"] += 1

                st.session_state.stats["response_time"] = response_time

                st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

# --- Clear chat ---
if st.button("🗑 Clear Chat", use_container_width=True):

    # Clear UI messages
    st.session_state.messages = []

    # Clear LangChain memory
    if session_id in st.session_state.store:
        del st.session_state.store[session_id]

    # Reset statistics
    st.session_state.stats = {
        "words": 0,
        "responses": 0,
        "response_time": 0
    }

    # Refresh app
    st.rerun()