from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
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

# Questions to ask:
# 1. What was microsoft first hardware product release?
# 2. how much did microsoft pay to acquire github
# 3. In which year did tesla begin production of the Roadster?
# 4. 