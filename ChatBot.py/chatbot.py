import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import time
import os
# Load environment variables
load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 AI Chatbot with LangChain & Hugging Face")
st.markdown("Talk to an AI-powered chatbot in real-time!")

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Custom chat container
chat_container = st.container()

# User input
with st.container():
    user_input = st.text_input("You:", "", key="placeholder", placeholder="Type your message here...")
    send_button = st.button("Send", use_container_width=True)

if send_button and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("AI is thinking..."):
        time.sleep(1)  # Simulate delay
        result = model.invoke(st.session_state.chat_history)
    
    st.session_state.chat_history.append({"role": "assistant", "content": result.content})
    st.rerun()

# Display chat history with styling
with chat_container:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"<div style='text-align: right; background-color:  #000000; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; background-color: #808080; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message['content']}</div>", unsafe_allow_html=True)

