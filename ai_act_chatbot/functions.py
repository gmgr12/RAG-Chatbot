import logging
import os
import pickle
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import faiss 
import transformers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
READER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FAISS_INDEX_PATH = "embeddings/knowledge_vector_database.faiss"

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

# Function to load the entire knowledge vector database from a file
def load_knowledge_vector_database(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            knowledge_vector_database = pickle.load(f)
        logging.info(f"Knowledge vector database loaded from {file_path}")
        return knowledge_vector_database
    else:
        logging.error(f"Knowledge vector database file {file_path} does not exist.")
        return None

# Function to retrieve relevant documents from the knowledge base
def retrieve_relevant_docs(query: str, knowledge_vector_database, k: int = 5):
    """
    Retrieve the most relevant documents from the FAISS knowledge base.
    
    Args:
        query (str): The user query.
        knowledge_vector_database: The FAISS knowledge base for retrieval.
        k (int): The number of top documents to retrieve.
    
    Returns:
        A tuple containing the retrieved documents, their combined text, and their metadata.
    """
    logging.info(f"Starting retrieval for query: {query}")
    retrieved_docs = knowledge_vector_database.similarity_search(query=query, k=k)

    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
    retrieved_docs_metadata = [doc.metadata for doc in retrieved_docs]
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {i}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)])

    return retrieved_docs, context, retrieved_docs_metadata

# Function to generate the final answer using the retrieved documents and LLM
def generate_answer_from_docs(query: str, context: str, reader_llm, tokenizer, retrieved_docs_metadata, max_new_tokens=500):
    """
    Generate an answer using the LLM based on the retrieved documents.

    Args:
        query (str): The user query.
        context (str): The text of the retrieved documents.
        reader_llm: The text generation pipeline (LLM).
        tokenizer: The tokenizer for formatting the chat-based prompt.
        retrieved_docs_metadata (list): Metadata of the retrieved documents.
        max_new_tokens (int): Maximum number of tokens for the generated answer.

    Returns:
        The generated answer from the LLM, including the page numbers of the retrieved chunks.
    """
    # Chat-style prompt for the model - prompt template
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Avoid referencing specific document names, such as "According to Document 0".  
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

    # Apply the chat template to the prompt
    rag_prompt_template = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

    # Generate the final answer
    generated_text = reader_llm(rag_prompt_template, truncation=True, max_new_tokens=max_new_tokens)

    # Process the generated text
    answer = generated_text[0]['generated_text']

    # Extract page numbers from metadata
    page_numbers = sorted(set([metadata['page'] for metadata in retrieved_docs_metadata]))
    page_numbers_str = ", ".join(map(str, page_numbers))

    # Append page numbers to the answer
    answer += f"\n\nPages retrieved from the document and included in the context for generating the answer: {page_numbers_str}"

    return answer

# Main script to load the knowledge vector database and test the complete process
if __name__ == "__main__":
    # Load the knowledge vector database
    knowledge_vector_database = load_knowledge_vector_database(FAISS_INDEX_PATH)
    if knowledge_vector_database is None:
        logging.error("Failed to load the knowledge vector database.")
        exit(1)

    # Initialize the reader model
    reader_llm, tokenizer = initialize_reader_model()

    # Define the query
    query = "What is the purpose of this Regulation?"

    # Retrieve relevant documents
    retrieved_docs, context, retrieved_docs_metadata = retrieve_relevant_docs(query, knowledge_vector_database)

    # Generate the answer
    answer = generate_answer_from_docs(query, context, reader_llm, tokenizer, retrieved_docs_metadata)
    print("Answer:", answer)