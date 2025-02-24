from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()

llm = HuggingFaceEndpoint(
 repo_id = 'google/gemma-2-2b-it',
 task = 'text-generation')

model = ChatHuggingFace(llm = llm)
result = model.invoke('who is elon musk')
print(result.content)
