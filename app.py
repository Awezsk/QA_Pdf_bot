import os
import streamlit as st
from pinecone import Pinecone, PineconeException
import cohere
from sentence_transformers import SentenceTransformer
import PyPDF2
import io
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone, Cohere, and SentenceTransformer
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
co = cohere.Client(st.secrets["COHERE_API_KEY"])
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to create embeddings
def create_embedding(text):
    try:
        return model.encode(text).tolist()
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        raise

# Function to load and preprocess the PDF document
def load_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        # Split the document into chunks (you may need to adjust this based on your document structure)
        chunks = text.split('\n\n')
        return chunks
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise

# Function to index the document in Pinecone
def index_document(chunks):
    index_name = "qa-bot-index"
    dimension = 384
    metric = "cosine"

    try:
        # Check if index already exists
        existing_indexes = pc.list_indexes()
        if index_name not in existing_indexes:
            st.write(f"Creating new index: {index_name}")
            # Define the index specification
            index_spec = {
                "name": index_name,  
                "metric": metric,
                "dimension": dimension,  
                "pod": {
                    "environment": str(os.environ)  
                }
            }
            pc.create_index(name=index_name, dimension=dimension, spec=index_spec)
        else:
            st.write(f"Index {index_name} already exists")

        # Connect to the index
        index = pc.Index(index_name)

        # Create and upsert vectors
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            vector = create_embedding(chunk)
            vectors_to_upsert.append((str(i), vector, {"text": chunk}))

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i+batch_size]
            try:
                index.upsert(vectors=batch)
            except PineconeException as e:
                logger.error(f"Pinecone upsert error: {str(e)}")
                raise

        st.success(f"Successfully indexed {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"Error in index_document: {str(e)}")
        st.error(f"Error indexing document: {str(e)}")

# Function to retrieve relevant chunks
def retrieve_chunks(query, top_k=3):
    try:
        index = pc.Index("qa-bot-index")
        query_vector = create_embedding(query)
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return [match['metadata']['text'] for match in results['matches']]
    except PineconeException as e:
        logger.error(f"Pinecone query error: {str(e)}")
        st.error(f"Error retrieving chunks: {str(e)}")
        return []

# Function to generate answer
def generate_answer(query, context):
    try:
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        response = co.generate(
            model='command-nightly',
            prompt=prompt,
            max_tokens=300,
            temperature=0.7,
            stop_sequences=["Question:"]
        )
        return response.generations[0].text.strip()
    except Exception as e:
        logger.error(f"Cohere generate error: {str(e)}")
        st.error(f"Error generating answer: {str(e)}")
        return ""

# Main QA function
def answer_question(query):
    relevant_chunks = retrieve_chunks(query)
    context = " ".join(relevant_chunks)
    answer = generate_answer(query, context)
    return answer, relevant_chunks

# Streamlit app
def main():
    st.title("Interactive QA Bot")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        # Process and index the document
        with st.spinner("Processing and indexing the document..."):
            chunks = load_pdf(uploaded_file)
            index_document(chunks)
            st.success("Document processed and indexed!")

        # Query input
        query = st.text_input("Ask a question about the document:")

        if query:
            with st.spinner("Generating answer..."):
                answer, relevant_chunks = answer_question(query)

            st.subheader("Answer:")
            st.write(answer)

            st.subheader("Relevant Document Segments:")
            for i, chunk in enumerate(relevant_chunks, 1):
                st.write(f"Segment {i}:")
                st.write(chunk)
                st.write("---")

if __name__ == "__main__":
    main()