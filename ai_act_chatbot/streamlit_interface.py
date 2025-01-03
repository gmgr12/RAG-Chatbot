import streamlit as st
from streamlit_chat import message
import os
import logging
from functions import (
    load_knowledge_vector_database,
    initialize_reader_model,
    retrieve_relevant_docs,
    generate_answer_from_docs,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the preprocessed knowledge vector database
FAISS_INDEX_PATH = "../embeddings/knowledge_vector_database.faiss"

# Initialize global variables
@st.cache_resource
def load_models():
    try:
        model, tokenizer = initialize_reader_model()
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        st.error("Failed to load models. Please check the logs.")
        return None, None  # Return None for both to avoid unpacking issues

@st.cache_resource
def load_knowledge_base():
    try:
        knowledge_vector_database = load_knowledge_vector_database(FAISS_INDEX_PATH)
        return knowledge_vector_database
    except Exception as e:
        logging.error(f"Error loading knowledge vector database: {e}")
        st.error("Failed to load knowledge vector database. Please check the logs.")
        return None

# Initialize the chatbot interface session state
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "knowledge_vector_database" not in st.session_state:
    st.session_state["knowledge_vector_database"] = None
if "model" not in st.session_state:
    st.session_state["model"] = None
if "tokenizer" not in st.session_state:
    st.session_state["tokenizer"] = None
if "document_loaded" not in st.session_state:
    st.session_state["document_loaded"] = False

# Load the knowledge vector database and models
if st.session_state["knowledge_vector_database"] is None:
    st.session_state["knowledge_vector_database"] = load_knowledge_base()
    st.session_state["model"], st.session_state["tokenizer"] = load_models()
    st.session_state["document_loaded"] = True  # Set document loaded flag

# Chatbot Interface
st.subheader("📄 Chat with the AI Act Document")

# Define user_input variable to avoid NameError
user_input = None

# Display question input only if the document is loaded
if st.session_state["document_loaded"]:
    user_input = st.text_input("Ask a question about the AI Act document:", disabled=False)
else:
    st.text_input("Loading the AI Act document. Please wait...", disabled=True)

# Ensure user input is taken only if document is loaded
if user_input and st.session_state["document_loaded"]:
    try:
        retrieved_docs, context, retrieved_docs_metadata = retrieve_relevant_docs(user_input, st.session_state["knowledge_vector_database"])
        answer = generate_answer_from_docs(user_input, context, st.session_state["model"], st.session_state["tokenizer"], retrieved_docs_metadata)

        # Display conversation history
        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer)
    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        st.error("An error occurred while processing your query.")

# Display chat history with question above the answer
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")  # Display user question
        message(st.session_state["generated"][i], key=str(i))  # Display bot answer