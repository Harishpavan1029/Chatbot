from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

# Initialize Hugging Face model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

# Wrap the Hugging Face model in LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Initialize embeddings for document vectorization
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB
vector_store = None

def process_document(file_path):
    global vector_store
    # Load document based on file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use .txt or .pdf")
    
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Create or update vector store
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    vector_store.persist()

def generate_response(query, history=[]):
    if vector_store is None:
        return "Please upload a document first."
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # Combine history and query for context
    context = " ".join([f"{msg['role']}: {msg['content']}" for msg in history]) + f" User: {query}"
    
    # Get response
    result = qa_chain({"query": context})
    response = result["result"]
    
    return response

def initialize_vector_store():
    global vector_store
    # Check if ChromaDB already exists
    if os.path.exists("./chroma_db"):
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)