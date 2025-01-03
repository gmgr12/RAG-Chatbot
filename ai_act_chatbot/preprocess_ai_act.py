import logging
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
PDF_FILE_PATH = "data/raw/TA-9-2024-0138_EN.pdf"
FAISS_INDEX_PATH = "embeddings/knowledge_vector_database.faiss"


# Function to split the document into chunks
def split_document_into_chunks(file_path: str, chunk_size: int, tokenizer_name: str = EMBEDDING_MODEL_NAME):
    """
    Load a document and split it into smaller chunks for processing.

    Args:
        file_path (str): Path to the document file.
        chunk_size (int): The maximum size of each chunk (number of tokens).
        tokenizer_name (str): The name of the tokenizer to use for splitting the document.

    Returns:
        List of split document chunks.
    """
    # Check if the document file exists
    if not os.path.isfile(file_path):
        logging.error(f"The file '{file_path}' does not exist.")
        return None

    # Load the document using PyPDFLoader
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    logging.info(f"The document has been loaded successfully. Total number of pages: {len(pages)}.")

    # Initialize a text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),  # 10% overlap between chunks
        add_start_index=True, 
        strip_whitespace=True
    )

    chunks = text_splitter.split_documents(pages)
    logging.info(f"The document has been split into {len(chunks)} chunks.")

    return chunks

# Function to generate embeddings for the document chunks
def generate_embeddings(chunks: list):
    """
    Generate embeddings for the given document chunks and store them using FAISS (uses the nearest neighbor search algorithm).
    
    Args:
        chunks (list): List of document chunks to generate embeddings for.
        
    Returns:
        FAISS index containing the document embeddings.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"},  # Use CPU for embeddings
        encode_kwargs={"normalize_embeddings": True}
    )
    logging.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    logging.info("Embeddings generated successfully.")

    return KNOWLEDGE_VECTOR_DATABASE

# Function to save the entire knowledge vector database to a file
def save_knowledge_vector_database(knowledge_vector_database, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(knowledge_vector_database, f)
    logging.info(f"Knowledge vector database saved to {file_path}")

# Main script
if __name__ == "__main__":
    # Split the document into chunks
    chunks = split_document_into_chunks(PDF_FILE_PATH, chunk_size=256)
    if chunks is not None:
        # Generate embeddings for the document chunks
        knowledge_vector_database = generate_embeddings(chunks)
        # Save the entire knowledge vector database to a file
        save_knowledge_vector_database(knowledge_vector_database, FAISS_INDEX_PATH)
    else:
        logging.error("Failed to split the document into chunks.")