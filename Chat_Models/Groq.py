from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
chat = ChatGroq(temperature=0.8, model_name="gemma2-9b-it", max_completion_tokens=50)
system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
result = chain.invoke({"text": ""})
print(result.content)