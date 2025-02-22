import os
import chromadb
from chromadb.utils import embedding_functions
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib  # For file hashing

# Constants
VECTOR_DB_DIR = "VectorDatabase"
COLLECTION_NAME = "pdf_collection"

# Initialize ChromaDB client (persistence handled automatically)
persist_directory = os.path.join(VECTOR_DB_DIR, "chroma_db")
client = chromadb.PersistentClient(path=persist_directory)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def get_file_hash(file_path):
    """Calculates the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(4096)  # Read in chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def load_and_embed_pdf(pdf_path, collection):
    """Loads, chunks, and embeds a PDF, checking for changes."""

    file_hash = get_file_hash(pdf_path)

    # Check if the PDF (by hash) is already in the collection
    results = collection.get(where={"source": pdf_path}, include=["metadatas"])
    if results['metadatas']:
        existing_metadata = results['metadatas'][0]
        if existing_metadata.get("file_hash") == file_hash:
            print(f"{pdf_path} (unchanged) already in the collection.")
            return  # Skip if file is unchanged

    # Load and chunk the PDF (same as before)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        page_number = chunk.metadata.get("page", None)
        file_name = os.path.basename(pdf_path)
        chunk_id = f"{file_name}:{page_number}:{i}" if page_number is not None else f"{file_name}:N/A:{i}"
        metadatas.append({"source": pdf_path, "chunk": i, "page": page_number, "file_hash": file_hash})  # Add file_hash
        ids.append(chunk_id)

    # If the PDF existed before, delete old entries first (to update changed content).
    if results['metadatas']:
        collection.delete(ids=results['ids']) # Delete by IDs

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    print(f"Added/Updated {len(chunks)} chunks from {pdf_path} to collection.")

def rebuild_database(pdf_directory, collection_name, collection):
    """Rebuilds the database if needed due to file changes."""
    files_in_db = set(meta['source'] for meta in collection.get(include=["metadatas"])['metadatas']) if collection.count()>0 else set()
    files_on_disk = set(os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf"))

    if files_in_db != files_on_disk:
        print("File changes detected. Rebuilding database...")
        client.delete_collection(name=collection_name) # Delete the collection
        collection = client.create_collection(name=collection_name, embedding_function=embedding_function) # Recreate it
        for pdf_path in files_on_disk:
            load_and_embed_pdf(pdf_path, collection)
    else:
        print("No file changes detected.")

def search_chroma(query, collection_name="pdf_collection", n_results=3):
    """Searches ChromaDB for the given query."""
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
    results = collection.query(query_texts=[query], n_results=n_results)
    return results

