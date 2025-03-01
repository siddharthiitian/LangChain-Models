import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import time

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 AI Chatbot with LangChain & Hugging Face")
st.markdown("Talk to an AI-powered chatbot in real-time!")

# Ask user for their Hugging Face API token
hf_token = st.text_input("Enter your Hugging Face API Token:", type="password")

if hf_token:
    # Initialize LLM only if the user provides an API token
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-2b-it",
        task="text-generation",
        huggingfacehub_api_token=hf_token
    )
    model = ChatHuggingFace(llm=llm)

    # Store chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Custom chat container
    chat_container = st.container()

    def send_message(user_input):
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with chat_container:
                placeholder = st.empty()
                with placeholder.container():
                    st.markdown(
                        "<div style='text-align: left; background-color: #E6E6E6; padding: 10px; border-radius: 10px; margin: 5px 0;'>🤖 AI is thinking...</div>",
                        unsafe_allow_html=True,
                    )
                
                time.sleep(1)  # Simulate delay
                result = model.invoke(st.session_state.chat_history)
            
            placeholder.markdown(
                f"<div style='text-align: left; background-color: #E6E6E6; padding: 10px; border-radius: 10px; margin: 5px 0;'>{result.content}</div>",
                unsafe_allow_html=True,
            )
            
            st.session_state.chat_history.append({"role": "assistant", "content": result.content})

    # User input
    with st.container():
        user_input = st.text_input("You:", "", key="input", placeholder="Type your message here...")
        send_button = st.button("Send", use_container_width=True)
        
        if send_button and user_input:
            send_message(user_input)

    # Display chat history with styling
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f"<div style='text-align: right; background-color: #000000; color: #FFFFFF; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align: left; background-color: #808080; color: #FFFFFF; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message['content']}</div>",
                    unsafe_allow_html=True,
                )
else:
    st.warning("⚠️ Please enter your Hugging Face API token to continue.")
