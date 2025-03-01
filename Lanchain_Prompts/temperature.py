from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
chat = ChatGroq(temperature=0.8, model_name="gemma2-9b-it")
system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
result = chain.invoke({"text": "Who is Elon Musk"})
print(result.content)