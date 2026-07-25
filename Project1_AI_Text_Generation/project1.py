import time
import streamlit as st
from langchain_groq import ChatGroq

st.set_page_config(page_title="AI Text Generation",layout="wide")
st.title("🤖 AI-Powered Text Generation")

with st.sidebar:
    api_key=st.text_input("Groq API Key",type="password")
    model=st.selectbox("Model",["llama-3.3-70b-versatile","llama-3.1-8b-instant"])
    temp=st.slider("Temperature",0.0,2.0,0.3)
    if st.button("Clear Chat"):
        st.session_state.messages=[]

if "messages" not in st.session_state:
    st.session_state.messages=[]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type your message...")

def groq_chat(msg):
    llm=ChatGroq(model=model,groq_api_key=api_key,temperature=temp)
    return llm.invoke(msg).content

if prompt:
    if not api_key:
        st.error("Enter Groq API key")
    elif not prompt.strip():
        st.warning("Enter a prompt")
    else:
        with st.spinner("Generating..."):
            ans=groq_chat(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})
        st.session_state.messages.append({"role":"assistant","content":ans})
        st.rerun()
