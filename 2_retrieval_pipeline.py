from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import getpass
import os
import truststore
truststore.inject_into_ssl()
from dotenv import load_dotenv
load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

db = Chroma(
    persist_directory= persistent_directory,
    embedding_function= embedding_model,
    collection_metadata= {"hnsw:space": "cosine"}
)
query = "how much did microsoft pay to acquire github"

retriever = db.as_retriever(search_kwargs = {"k":5})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("--- Context ---")

for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}: \n {doc.page_content} \n")


combined_input = f"""Based on the following documents, please answer this question: {query}
Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
please provide clear, helpful answer using only infomation from these documents. If you can't find the answer just say you have not found the answer
 """
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    
model = ChatGoogleGenerativeAI(model = "gemini-3.8-flash")

messages = [
    SystemMessage(content = "You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("\n ---Generated Response ---")

print(f"Content only \n{result.text}")



# Questions to ask:
# 1. What was microsoft first hardware product release?
# 2. how much did microsoft pay to acquire github
# 3. In which year did tesla begin production of the Roadster?
# 4. 