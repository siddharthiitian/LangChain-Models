# import streamlit as st

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import os
load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
 repo_id = 'google/gemma-2-2b-it',
 task = 'text-generation')

model = ChatHuggingFace(llm = llm)
Chat_history=[]
while True:
  user_input  = input("You: ")
  Chat_history.append({"role": "user", "content": user_input})
  if user_input == 'exit':
      break
  result = model.invoke(Chat_history)
  Chat_history.append({"role": "assistant", "content": result.content})
  print('AI:',result.content)

print(Chat_history)
