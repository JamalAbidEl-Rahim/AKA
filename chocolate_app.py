import streamlit as st
import numpy as np
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


class PDFVectorDB:
    def __init__(self, pdf_path):
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)

        # Initialize embeddings with new import
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)

    def search(self, query, k=8):
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def as_retriever(self, search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 2}
        return self


def retrieve_context_per_question(query, retriever):
    return retriever.search(query)


def show_context(context):
    return "\n\n".join(context)


def get_mistral_response(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        return response.json()['response']
    except Exception as e:
        return f"Error communicating with Mistral: {e}"


# Streamlit UI
st.set_page_config(page_title="Book Knowledge Bot", page_icon="📚")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize VectorDB
if 'vector_db' not in st.session_state:
    pdf_path = "/Users/basilyassin/Desktop/streamlit_mistral/history.pdf"
    st.session_state.vector_db = PDFVectorDB(pdf_path)
    st.session_state.retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 2})

st.title("📚 Book Knowledge Bot")
st.write("Ask me anything about the book!")

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("What would you like to know about the book?"):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Get relevant information and generate response using retriever
    context = retrieve_context_per_question(prompt, st.session_state.retriever)
    context_text = show_context(context)
    full_prompt = f"Using this information from the book: {context_text}\n\nQuestion: {prompt}\nAnswer:"

    # Get and display Mistral's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_mistral_response(full_prompt)
            st.write(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write("""
    This chatbot uses:
    - LangChain for PDF processing
    - FAISS for vector storage
    - HuggingFace embeddings
    - Mistral LLM for generating responses
    - Knowledge from the book
    """)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Test retriever (optional)
if st.checkbox("Show Test Query"):
    test_query = "Who edited the book 'Beckett's Industrial Chocolate Manufacture and Use'"
    test_context = retrieve_context_per_question(test_query, st.session_state.retriever)
    st.write("Test Query Results:")
    st.write(show_context(test_context))