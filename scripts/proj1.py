from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import cohere

# 1. Split the document

loader = PyPDFLoader("data/raw/TA-9-2024-0138_EN.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

# 2. Generate and store the embeddings

client = chromadb.Client()
collection = client.create_collection(name="pdf_embeddings")

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = [model.encode(chunk.page_content) for chunk in split_docs]

for i, embedding in enumerate(embeddings):
        metadata = {
            "source": split_docs[i].metadata['source'],
            "page": split_docs[i].metadata['page']
        }
        collection.add(
            ids=[f"doc_chunk_{i}"], 
            embeddings=[embedding], 
             metadatas=[metadata]
        )

# 3. Create a retriever

query = "Who the Commission shall consult before adopting a delegated act?"
results = collection.query(
        query_embeddings=[model.encode(query)],
        n_results=5
    )

# 4. Prompt template

context = ""

for i, doc_id in enumerate(results['ids'][0]):
    chunk_content = split_docs[int(doc_id.split('_')[2])].page_content
    context += f"\nChunk {i+1}: {chunk_content}\n"
    print(f"Chunk ID: {doc_id}")
    print(f"Distance: {results['distances'][0][i]}")
    print(f"Chunk Content: {chunk_content}")
    print("---------")

prompt_template = f"""
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {query}.
Provide a detailed answer.
Don’t justify your answers.
Don’t give information not mentioned in the CONTEXT INFORMATION.
Do not say "according to the context" or "mentioned in the context" or similar.
"""
print(prompt_template)

co = cohere.Client('Fcu5GXKilkxwAfdA85stnMI0mfU7Me5oLFDhyi5I')
response = co.chat(
  message=prompt_template,
  model="command",
  temperature=0.3
)
print(response.text)