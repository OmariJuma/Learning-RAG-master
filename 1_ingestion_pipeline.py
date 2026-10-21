# pip install langchain langchain-community langchain_text_splitters langchain_openai langchain_chroma python_dotenv
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings

from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

def load_documents(docs_path = "docs"):
    print(f"Loading documents from {docs_path}")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")

    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader,
        loader_kwargs={"encoding" : "utf-8"},
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company files.")
   
    print(f"{len(documents)} documents loaded")
    for i, doc in enumerate(documents[:2]): # show first 2
        print(f"\n Document {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")
        return documents


def split_documents(documents, chunk_size = 1000, chunk_overlap = 0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n --- Chunk {i+1} --- ")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"*50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) -5} more chunks ")
    return chunks


def create_vector_store(chunks, persist_directory = "db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in chromadb...")
    # embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents= chunks,
        embedding= embedding_model,
        persist_directory=  persist_directory,
        collection_metadata= {"hnsw:space": "cosine"}

    ) 
    print("--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    """This is the main method"""
    # 1. Load the documents
    documents = load_documents()

    # 2. Chunk the documents
    chunks = split_documents(documents)

    # 3. Embeddings and storage in vector db
    create_vector_store(chunks)


if __name__ == "__main__":
    main()

