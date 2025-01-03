import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoModelForCausalLM, pipeline
import faiss 
import transformers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


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

# Function to initialize the reader model
def initialize_reader_model(model_name: str = READER_MODEL_NAME):
    """
    Initialize the LLM model for text generation.
    
    Args:
        model_name (str): The name of the model to use for the LLM.
    
    Returns:
        A HuggingFace pipeline for text generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, device_map="auto", torch_dtype="auto")    

    reader_llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer, 
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.2,
    return_full_text=False,
    )
    logging.info(f"Reader LLM model '{model_name}' initialized successfully.")
    return reader_llm, tokenizer

# Function to retrieve relevant documents from the knowledge base
def retrieve_relevant_docs(query: str, knowledge_vector_database, k: int = 5):
    """
    Retrieve the most relevant documents from the FAISS knowledge base.
    
    Args:
        query (str): The user query.
        knowledge_vector_database: The FAISS knowledge base for retrieval.
        k (int): The number of top documents to retrieve.
    
    Returns:
        A tuple containing the retrieved documents and their combined text.
    """
    logging.info(f"Starting retrieval for query: {query}")
    retrieved_docs = knowledge_vector_database.similarity_search(query=query, k=k)

    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {i}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)])

    return retrieved_docs, context

# Function to generate the final answer using the retrieved documents and LLM
def generate_answer_from_docs(query: str, context: str, reader_llm, tokenizer, max_new_tokens=512):
    """
    Generate an answer using the LLM based on the retrieved documents.

    Args:
        query (str): The user query.
        context (str): The text of the retrieved documents.
        reader_llm: The text generation pipeline (LLM).
        tokenizer: The tokenizer for formatting the chat-based prompt.
        max_new_tokens (int): Maximum number of tokens for the generated answer.

    Returns:
        The generated answer from the LLM.
    """
    # Chat-style prompt for the model
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
        {
            "role": "user",
            "content": f"""Context:
            {context}
            ---
            Now here is the question you need to answer:

            Question: {query}"""
        },
    ]

    # Apply the chat-style template using tokenizer (if needed)
    rag_prompt_template = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    # Generate the final answer
    generated_text = reader_llm(rag_prompt_template, truncation=True, max_new_tokens=max_new_tokens)

    # Process the generated text
    answer = generated_text[0]['generated_text']

    return answer