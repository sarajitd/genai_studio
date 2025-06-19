from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

def create_vectorstore(data_path="C:/Users/saraj/genai_studio/my_env/data/my_dataset.txt", index_path="C:/Users/saraj/genai_studio/my_env/data/index.faiss"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Read your dataset
    with open(data_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]
    
    # Convert data into FAISS-compatible format
    docs = [Document(page_content=text) for text in texts]
    
    # Create and save FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(index_path)

    return vectorstore

def load_vectorstore(index_path="data/index.faiss"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore





