from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from chromadb import PersistentClient

import os
import json

# Load configurations from config.json
def load_config(config_path=os.path.join(os.path.dirname(__file__), 'config.json')):
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return None
    with open(config_path, 'r') as file:
        return json.load(file)
    
config = load_config()

# Configuration
if config is None:
    exit()

MODEL_NAME = config['embedding_model_name']
PERSIST_DIRECTORY = config['vector_database_path']
OPENAI_API_KEY = config['openai_api_key']
TEMPERATURE = config.get('temperature', 0.0) #Default 0.0, but can be overriden in config
MAX_TOKENS = config.get('max_tokens', 150)
MODEL_LLM = config.get('llm_model_name', "gpt-3.5-turbo") #Default to gpt-3.5-turbo
COLLECTION_NAME = config['collection_name']
CHUNK_SIZE = config.get('chunk_size', 1000)
CHUNK_OVERLAP = config.get('chunk_overlap', 200)
DATA_PATH = config.get('data_path', "./data")

#Initialize components
#Using Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
#Load chroma vector db
persist_directory = PERSIST_DIRECTORY
client = PersistentClient(path=persist_directory)
vectordb = Chroma(client=client, embedding_function=embeddings)
#Load OpenAI LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=MODEL_LLM, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
#Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# Define a prompt template for RAG
template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)

# RAG Chain
def rag_chain():
  retriever = vectordb.as_retriever(search_kwargs={"k": 5})

  rag_chain_instance = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
  return rag_chain_instance


def ingest_documents(data_path):
    """Ingests documents and saves it to chroma db"""
    loader = DirectoryLoader(data_path, glob="**/*.txt", show_progress=True)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    vectordb.add_documents(texts)

if __name__ == '__main__':
    # Load Config
    config = load_config()
    if config is None:
            exit()

    # Ingest Documents
    ingest_documents(DATA_PATH)

    # Test RAG pipeline
    test_query = "What is the purpose of this module?"
    rag = rag_chain()
    response = rag.invoke(test_query)
    print(f"Query: {test_query}")
    print(f"Response: {response}")
