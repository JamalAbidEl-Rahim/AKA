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

    def vector_search(self, query: str, k: int = 5) -> List[str]:
        """
        Perform vector-based search only.
        """
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            return []

    def bm25_search(self, query: str, k: int = 5) -> List[str]:
        """
        Perform BM25 keyword search only.
        """
        try:
            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)

            # Get all documents from vector store (we need them for reference)
            all_docs = self.vector_store.similarity_search("", k=1000)  # Get as many as possible

            # Sort by BM25 score
            sorted_indices = np.argsort(bm25_scores)[::-1]

            # Return top k
            return [all_docs[i].page_content for i in sorted_indices[:k] if i < len(all_docs)]
        except Exception as e:
            print(f"Error in BM25 search: {str(e)}")
            return []


def get_mistral_response(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        return response.json()['response']
    except Exception as e:
        return f"Error communicating with Mistral: {e}"


def agent_retrieve_answer(query, vector_db):
    """
    Implements an Agent RAG approach that makes decisions about retrieval strategy
    and dynamically plans the reasoning approach based on the query type.

    Args:
        query (str): The user's question
        vector_db (FusionPDFVectorDB): The vector database with different search methods

    Returns:
        str: The final answer after agent reasoning
    """
    try:
        # Step 1: Analyze the query to determine the best retrieval strategy
        query_analysis_prompt = f"""
        Analyze this query and determine the best search strategy. Your options are:
        1. Vector search - best for conceptual questions and semantic understanding
        2. BM25 search - best for factual recall and when specific terms must be matched
        3. Fusion search - best for balanced queries needing both semantic and keyword matching

        Query: "{query}"

        Think step by step about what kind of information is needed. Respond with ONLY ONE of these exact terms: "Vector", "BM25", or "Fusion".
        """

        search_strategy = get_mistral_response(query_analysis_prompt).strip()

        # Ensure we get a valid strategy, default to Fusion if response is unclear
        valid_strategies = ["Vector", "BM25", "Fusion"]
        if search_strategy not in valid_strategies:
            search_strategy = "Fusion"

        # Step 2: Determine k (number of contexts) based on query complexity
        k_determination_prompt = f"""
        Analyze this query and determine how many document chunks we should retrieve.
        For simple factual queries, fewer chunks (2-3) are needed.
        For complex analytical questions, more chunks (5-8) are better.

        Query: "{query}"

        Analyze step by step what kind of information is needed, then respond with ONLY a single number between 2 and 8.
        """

        try:
            k_value = int(get_mistral_response(k_determination_prompt).strip())
            # Ensure k is within bounds
            k_value = max(2, min(8, k_value))
        except:
            # Default to 5 if we can't parse a valid number
            k_value = 5

        # Step 3: Retrieve contexts using the determined strategy
        if search_strategy == "Vector":
            contexts = vector_db.vector_search(query, k=k_value)
            retrieval_method = "Vector Search (semantic matching)"
        elif search_strategy == "BM25":
            contexts = vector_db.bm25_search(query, k=k_value)
            retrieval_method = "BM25 Search (keyword matching)"
        else:  # Fusion
            alpha_value = 0.5  # Could be dynamically determined as well
            contexts = vector_db.fusion_search(query, k=k_value, alpha=alpha_value)
            retrieval_method = "Fusion Search (balanced semantic and keyword matching)"

        # Step 4: Analyze if we need additional contexts
        if len(contexts) > 0:
            first_pass_context = "\n\n".join(contexts)
            sufficiency_prompt = f"""
            Given this information retrieved from the document:
            {first_pass_context}

            And this query: "{query}"

            Do we have enough information to fully answer the query? Respond with ONLY "YES" or "NO".
            """

            sufficiency_response = get_mistral_response(sufficiency_prompt).strip().upper()

            # If we need more information and haven't maxed out our context retrieval
            if sufficiency_response == "NO" and k_value < 8:
                # Try a different retrieval method to get complementary information
                if search_strategy != "Fusion":
                    additional_contexts = vector_db.fusion_search(query, k=3)
                    contexts.extend(additional_contexts)
                    retrieval_method += " + Additional Fusion Search"
                else:
                    # If we already used fusion, try pure vector for additional context
                    additional_contexts = vector_db.vector_search(query, k=3)
                    contexts.extend(additional_contexts)
                    retrieval_method += " + Additional Vector Search"

        # Step 5: Determine the reasoning approach needed
        reasoning_prompt = f"""
        Given this query: "{query}"

        What kind of reasoning approach should I use to answer it effectively?
        1. Direct factual answer - for simple factual questions
        2. Comparative analysis - for questions comparing multiple items
        3. Causal reasoning - for questions about why something happened
        4. Step-by-step explanation - for process or how-to questions
        5. Multi-perspective synthesis - for complex questions needing different viewpoints

        Think step by step, then respond with ONLY ONE number (1-5) that best matches the reasoning approach needed.
        """

        try:
            reasoning_type = int(get_mistral_response(reasoning_prompt).strip())
            # Ensure reasoning_type is within bounds
            reasoning_type = max(1, min(5, reasoning_type))
        except:
            # Default to 1 if we can't parse a valid number
            reasoning_type = 1

        # Map reasoning type to approach instruction
        reasoning_approaches = {
            1: "Provide a direct and concise factual answer based on the information.",
            2: "Compare and contrast the relevant elements to answer the question.",
            3: "Explain the causal relationships and why things happened this way.",
            4: "Break down the explanation into clear sequential steps.",
            5: "Synthesize multiple perspectives to provide a comprehensive answer."
        }

        reasoning_instruction = reasoning_approaches.get(reasoning_type, reasoning_approaches[1])

        # Step 6: Generate the final answer
        context_text = "\n\n".join(contexts)

        final_prompt = f"""
        Using this information retrieved from the document (via {retrieval_method}): 

        {context_text}

        Question: {query}

        {reasoning_instruction}

        Answer the question thoroughly based on the provided information. If you cannot answer based on the provided context, say so clearly.
        """

        final_answer = get_mistral_response(final_prompt)

        # Store the agent's thought process (can be displayed in debug mode)
        agent_thought_process = {
            "search_strategy": search_strategy,
            "k_value": k_value,
            "retrieval_method": retrieval_method,
            "reasoning_approach": reasoning_approaches[reasoning_type],
            "num_contexts": len(contexts)
        }

        return final_answer, agent_thought_process

    except Exception as e:
        return f"Error in Agent RAG process: {str(e)}\n\nFalling back to basic retrieval. Please try again with a reformulated question.", None


# Streamlit UI
st.set_page_config(page_title="Book Knowledge Bot", page_icon="📚", layout="wide")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize RAG type if not set
if 'rag_type' not in st.session_state:
    st.session_state.rag_type = "Fusion RAG"

# Initialize fusion search type if not set
if 'fusion_search_type' not in st.session_state:
    st.session_state.fusion_search_type = "Fusion"

# Initialize debug mode
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Initialize agent thought process store
if 'agent_thoughts' not in st.session_state:
    st.session_state.agent_thoughts = {}

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

# Create a layout with two main containers
# Top container for title and messages
main_container = st.container()
# Bottom container for input that will always be at the bottom
input_container = st.container()

# Use the main container for the title and chat history
with main_container:
    st.title("📚 Book Knowledge Bot")
    st.write("Ask me anything about the book!")

    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show agent thought process in debug mode
            if st.session_state.debug_mode and message["role"] == "assistant" and "message_id" in message:
                if message["message_id"] in st.session_state.agent_thoughts:
                    with st.expander("Agent Thought Process"):
                        thought_process = st.session_state.agent_thoughts[message["message_id"]]
                        st.json(thought_process)

# Use a container at the bottom of the page for inputs
with input_container:
    # Create columns for RAG type, search type dropdown, and input
    col0, col1, col2 = st.columns([1, 1, 3])

    with col0:
        rag_type = st.selectbox(
            "RAG Type",
            ["Fusion RAG", "Agent RAG"],
            index=0 if st.session_state.rag_type == "Fusion RAG" else 1,
            label_visibility="collapsed",
            key="rag_type_select"
        )
        # Update session state when changed
        if rag_type != st.session_state.rag_type:
            st.session_state.rag_type = rag_type
            # Force a rerun to update the sidebar
            st.rerun()

    # Only show search type dropdown for Fusion RAG
    with col1:
        if st.session_state.rag_type == "Fusion RAG":
            search_type = st.selectbox(
                "Search Type",
                ["Fusion", "Vector", "BM25"],
                index=["Fusion", "Vector", "BM25"].index(st.session_state.fusion_search_type),
                label_visibility="collapsed",
                key="search_type_select"
            )
            # Update session state
            st.session_state.fusion_search_type = search_type
        else:
            # Placeholder for consistency in layout
            st.empty()

    with col2:
        prompt = st.chat_input("What would you like to know about the book?")

# User input processing
if prompt:
    import uuid

    message_id = str(uuid.uuid4())

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Process based on RAG type
    with st.spinner(""):
        try:
            if st.session_state.rag_type == "Fusion RAG":
                # Get parameters from sidebar
                k_value = st.session_state.get('k_value', 5)
                alpha_value = st.session_state.get('alpha_value', 0.5)

                # Get context based on search type
                search_type = st.session_state.fusion_search_type
                if search_type == "Fusion":
                    context = st.session_state.vector_db.fusion_search(prompt, k=k_value, alpha=alpha_value)
                    search_method = "Fusion Search (Vector + BM25)"
                elif search_type == "Vector":
                    context = st.session_state.vector_db.vector_search(prompt, k=k_value)
                    search_method = "Vector Search"
                else:  # BM25
                    context = st.session_state.vector_db.bm25_search(prompt, k=k_value)
                    search_method = "BM25 Keyword Search"

                # Format context and prepare prompt
                context_text = "\n\n".join(context)
                full_prompt = f"Using this information from the book (retrieved via {search_method}): {context_text}\n\nQuestion: {prompt}\nAnswer:"

                # Get and display response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_mistral_response(full_prompt)
                        st.write(response)

                # Store in chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "message_id": message_id
                })
            else:
                # Agent RAG implementation
                with st.chat_message("assistant"):
                    with st.spinner("Agent thinking..."):
                        response, agent_thoughts = agent_retrieve_answer(prompt, st.session_state.vector_db)
                        st.write(response)

                        # Show debug information if enabled
                        if st.session_state.debug_mode and agent_thoughts:
                            with st.expander("Agent Thought Process"):
                                st.json(agent_thoughts)

                # Store agent thoughts for this message
                if agent_thoughts:
                    st.session_state.agent_thoughts[message_id] = agent_thoughts

                # Store in chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "message_id": message_id
                })

        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# Sidebar with additional information - dynamically changes based on RAG type
with st.sidebar:
    st.header("About")

    if st.session_state.rag_type == "Fusion RAG":
        st.write("""
        **Fusion RAG** uses multiple retrieval methods to find the most relevant information:
        - **Fusion**: Combines vector and keyword search for balanced results
        - **Vector**: Semantic search using embeddings for conceptual matching
        - **BM25**: Keyword search focused on exact term matching
        """)

        # Search parameters for Fusion RAG
        st.subheader("Search Parameters")
        k_value = st.slider("Number of contexts (k)", min_value=1, max_value=10, value=5, key="k_value")
        alpha_value = st.slider("Vector search weight (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                key="alpha_value")

        st.info(f"Currently using: {st.session_state.fusion_search_type} search")
    else:
        st.write("""
        **Agent RAG** uses an intelligent agent to:
        - Automatically select the optimal retrieval method
        - Plan multi-step reasoning approaches
        - Determine when to retrieve more information
        - Generate comprehensive answers using all available context
        """)

        # Agent parameters
        st.subheader("Agent Parameters")
        st.slider("Thinking steps", min_value=1, max_value=5, value=3, key="agent_thinking_steps",
                  help="Controls how thoroughly the agent analyzes the query")
        st.slider("Reasoning depth", min_value=1, max_value=5, value=3, key="agent_reasoning_depth",
                  help="Controls how detailed the agent's reasoning will be")

        # Debug mode toggle
        st.checkbox("Debug mode", value=st.session_state.debug_mode, key="debug_mode_toggle",
                    on_change=lambda: setattr(st.session_state, "debug_mode", st.session_state.debug_mode_toggle))

        if st.session_state.debug_mode:
            st.info("Debug mode enabled - agent thinking process will be shown")
        else:
            st.info("Agent RAG active")

    # Common controls
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.agent_thoughts = {}
        st.rerun()