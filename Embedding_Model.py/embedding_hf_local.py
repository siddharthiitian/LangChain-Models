from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
document = ['Who is elon musk','What is capital of india','What is your name']
vector = embedding.embed_documents(document)
print(vector)#384 length