from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi
import faiss
import numpy as np


# Step 1: Load PDF
def load_pdf(path):
    reader = PdfReader(path)
    return [page.extract_text() for page in reader.pages]


# Step 2: Configure chunking
def configure_chunking():
    chunk_size = 1000  # Set chunk size
    chunk_overlap = 200  # Set chunk overlap
    return chunk_size, chunk_overlap


# Step 3: Create embeddings using SentenceTransformers
def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=False)
    return np.array(embeddings, dtype="float32")


# Step 4: Build FAISS Index
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Get the dimensionality of the embeddings
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
    index.add(embeddings)  # Add the embeddings to the index
    return index


# Step 5: Create BM25 Index
def create_bm25_index(texts, chunk_size, chunk_overlap):
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i:i + chunk_size])
    tokenized_chunks = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks), chunks


# Step 6: Fusion Retrieval
def fusion_retrieve(query, vectorstore, bm25, texts, query_embedding, k=5, alpha=0.5):
    # BM25 scores
    bm25_scores = bm25.get_scores(query.split())

    # Vector similarity scores
    _, vector_indices = vectorstore.search(query_embedding, len(texts))
    vector_scores = 1 - np.array([bm25_scores[idx] for idx in vector_indices[0]])

    # Normalize scores and combine
    bm25_scores = (bm25_scores - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores))
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    return [texts[idx] for idx in sorted_indices[:k]]


# Step 7: Main function
def main():
    print("Welcome to the Interactive Retrieval System!")

    # Set PDF path here
    pdf_path = "your_file.pdf"

    # Load PDF
    print("Loading PDF...")
    texts = load_pdf(pdf_path)

    # Configure chunking
    chunk_size, chunk_overlap = configure_chunking()

    # Generate embeddings and FAISS index
    print("Creating embeddings using sentence-transformers...")
    embeddings = create_embeddings(texts)
    vectorstore = build_faiss_index(embeddings)

    # Create BM25 index
    print("Creating BM25 index...")
    bm25, chunks = create_bm25_index(texts, chunk_size, chunk_overlap)

    # Perform Retrieval
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting. Thank you!")
            break

        print("Select retrieval method:")
        print("1. BM25")
        print("2. Vector-based retrieval")
        print("3. Fusion retrieval (BM25 + Vector)")
        retrieval_choice = int(input("Enter your choice: "))

        if retrieval_choice == 1:
            print("Performing BM25 retrieval...")
            results = bm25.get_top_n(query.split(), chunks, n=5)
        elif retrieval_choice == 2:
            print("Performing vector-based retrieval...")
            query_embedding = create_embeddings([query])
            _, indices = vectorstore.search(query_embedding, 5)
            results = [chunks[i] for i in indices[0]]
        elif retrieval_choice == 3:
            print("Performing fusion retrieval...")
            alpha = float(input("Enter the alpha value (weight for vector scores, 0 to 1): "))
            query_embedding = create_embeddings([query])
            results = fusion_retrieve(query, vectorstore, bm25, chunks, query_embedding, k=5, alpha=alpha)
        else:
            print("Invalid choice. Try again.")
            continue

        print("\nTop Results:")
        for idx, result in enumerate(results, 1):
            print(f"{idx}. {result}\n")


if __name__ == "__main__":
    main()
