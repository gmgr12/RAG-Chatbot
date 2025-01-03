import streamlit as st
from streamlit_chat import message
import os
import logging
from functions import (
    split_document_into_chunks,
    generate_embeddings,
    initialize_reader_model,
    retrieve_relevant_docs,
    generate_answer_from_docs,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def initialize_knowledge_base(file_path):
    chunks = split_document_into_chunks(file_path, chunk_size=256)
    knowledge_vector_database = generate_embeddings(chunks)
    return knowledge_vector_database

# Streamlit page setup
st.set_page_config(page_title="Document-Based Chatbot", layout="centered")

st.title("📄 Document-Based Chatbot")

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF document", type="pdf")

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

# Check if a document is uploaded
if uploaded_file is not None:
    os.makedirs("uploaded_documents", exist_ok=True)

    # Save uploaded file temporarily
    with open(os.path.join("uploaded_documents", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the document: split it into chunks and generate embeddings
    st.session_state["knowledge_vector_database"] = initialize_knowledge_base(
        os.path.join("uploaded_documents", uploaded_file.name)
    )
    st.success(f"Document '{uploaded_file.name}' processed and loaded successfully.")

    # Load the models (Reader LLM and Tokenizer)
    st.session_state["model"], st.session_state["tokenizer"] = load_models()
    st.session_state["document_loaded"] = True  # Set document loaded flag

# Chatbot Interface
st.subheader("Chat with the Document")

# Define user_input variable to avoid NameError
user_input = None

# Display question input only if the document is loaded
if st.session_state["document_loaded"]:
    user_input = st.text_input("Ask a question about the document:", disabled=False)
else:
    st.text_input("Upload a document to enable asking questions.", disabled=True)

# Ensure user input is taken only if document is loaded
if user_input and st.session_state["document_loaded"]:
    try:
        retrieved_docs, context = retrieve_relevant_docs(user_input, st.session_state["knowledge_vector_database"])
        answer = generate_answer_from_docs(user_input, context, st.session_state["model"], st.session_state["tokenizer"])

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
