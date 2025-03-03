import streamlit as st
import numpy as np
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from typing import List
from langchain.docstore.document import Document
import os
import pickle


class FusionPDFVectorDB:
    def __init__(self, pdf_path, vector_store_path):
        """
        Initialize with either a new PDF or existing vector store
        """
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        try:
            if os.path.exists(vector_store_path):
                print("Loading existing vector store...")
                # Load existing vector store
                self.vector_store = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )

                # Test the vector store with a simple query
                test_embedding = self.embeddings.embed_query("test")
                self.vector_store.index.search(np.array([test_embedding]), k=1)

                # Get all documents for BM25
                print("Loading documents for BM25...")
                all_docs = self.vector_store.similarity_search("test query", k=5)
                self.bm25 = self.create_bm25_index(all_docs)
                print("Vector store loaded successfully!")

            else:
                print("Creating new vector store from PDF...")
                # Create new vector store from PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                texts = text_splitter.split_documents(documents)

                # Create vector store
                self.vector_store = FAISS.from_documents(texts, self.embeddings)

                # Create BM25 index
                self.bm25 = self.create_bm25_index(texts)

                # Save vector store
                os.makedirs(vector_store_path, exist_ok=True)
                self.vector_store.save_local(vector_store_path)
                print("New vector store created and saved!")

        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # If loading fails, try to create a new one
            print("Attempting to create new vector store...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            texts = text_splitter.split_documents(documents)

            # Create vector store
            self.vector_store = FAISS.from_documents(texts, self.embeddings)

            # Create BM25 index
            self.bm25 = self.create_bm25_index(texts)

            # Save vector store
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store.save_local(vector_store_path)
            print("New vector store created and saved!")

    def create_bm25_index(self, documents: List[Document]) -> BM25Okapi:
        """Create a BM25 index from the given documents."""
        tokenized_docs = [doc.page_content.split() for doc in documents]
        return BM25Okapi(tokenized_docs)

    def fusion_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[str]:
        """
        Perform fusion retrieval combining keyword-based (BM25) and vector-based search.
        """
        try:
            epsilon = 1e-8

            # Get vector search results
            vector_results = self.vector_store.similarity_search_with_score(query, k=k)
            vector_docs = [doc for doc, _ in vector_results]
            vector_scores = np.array([score for _, score in vector_results])

            # Get BM25 scores for the same documents
            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)

            # Normalize scores
            if len(vector_scores) > 0:
                vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (
                            np.max(vector_scores) - np.min(vector_scores) + epsilon)

            if len(bm25_scores) > 0:
                bm25_scores = (bm25_scores - np.min(bm25_scores)) / (
                            np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

            # Combine scores (use only vector scores if BM25 fails)
            if len(bm25_scores) == len(vector_scores):
                combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores[:len(vector_scores)]
            else:
                combined_scores = vector_scores

            # Rank and return top k documents
            sorted_indices = np.argsort(combined_scores)[::-1]
            return [vector_docs[i].page_content for i in sorted_indices[:k]]
        except Exception as e:
            print(f"Error in fusion search: {str(e)}")
            # Fallback to simple vector search
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]


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

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize VectorDB with fusion search
if 'vector_db' not in st.session_state:
    pdf_path = "/Users/basilyassin/Desktop/streamlit_mistral/Aka Book.pdf"
    vector_store_path = "/Users/basilyassin/Desktop/streamlit_mistral/vector_stores/detailed_store"

    with st.spinner("Loading vector store..."):
        try:
            st.session_state.vector_db = FusionPDFVectorDB(pdf_path, vector_store_path)
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            st.stop()

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

    # Get relevant information using fusion search
    with st.spinner(""):
        try:
            context = st.session_state.vector_db.fusion_search(prompt, k=5, alpha=0.5)
            context_text = "\n\n".join(context)
            full_prompt = f"Using this information from the book: {context_text}\n\nQuestion: {prompt}\nAnswer:"

            # Get and display Mistral's response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_mistral_response(full_prompt)
                    st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Sidebar with additional information
with st.sidebar:
    st.header("About")
    st.write("""
    This chatbot uses:
    - LangChain for PDF processing
    - FAISS for vector storage
    - BM25 for keyword search
    - Fusion retrieval combining vector and keyword search
    - HuggingFace embeddings (all-MiniLM-L6-v2)
    - Mistral LLM for generating responses
    """)

    # Add sliders for fusion parameters
    st.subheader("Search Parameters")
    k_value = st.slider("Number of contexts (k)", min_value=1, max_value=10, value=5)
    alpha_value = st.slider("Vector search weight (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()