from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

chat = ChatGoogleGenerativeAI(temperature=0.8, model="gemini-1.5-pro", max_completion_tokens=50)
result = chat.invoke('Give some abusive word')
print(result.content)