import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import time

st.set_page_config(page_title="Travel Itinerary Planner", page_icon="🌍", layout="centered")
st.title("🌍 AI Travel Itinerary Planner")
st.markdown("Plan your perfect trip with AI!")

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

    def generate_itinerary(location, days, user_type):
        prompt = (
            f"Generate a detailed travel itinerary for a trip to {location}. "
            f"The trip duration is {days} days. The traveler is a {user_type}. "
            f"Provide day-wise plans, including attractions, activities, and food recommendations."
        )
        
        with st.spinner("🛫 Generating your travel itinerary..."):
            time.sleep(1)  # Simulate delay
            result = model.invoke(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": result.content})

    # User inputs
    location = st.text_input("Where are you traveling to?", "")
    days = st.number_input("How many days are you staying?", min_value=1, step=1)
    user_type = st.selectbox("What kind of traveler are you?", ["Adventurer", "Relaxer", "Foodie", "Culture Seeker"])
    plan_button = st.button("Generate Itinerary", use_container_width=True)
    
    if plan_button and location.strip():
        generate_itinerary(location.strip(), days, user_type)
    
    # Display chat history with styling
    with chat_container:
        for message in st.session_state.chat_history:
            st.markdown(
                f"<div style='text-align: left; background-color: #808080; color: #FFFFFF; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message['content']}</div>",
                unsafe_allow_html=True,
            )
else:
    st.warning("⚠️ Please enter your Hugging Face API token to continue.")
