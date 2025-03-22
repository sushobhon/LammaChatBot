# Import Necessary Libraries
import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants
VECTOR_DB_DIR = "VectorDatabase"
COLLECTION_NAME = "pdf_collection"

# Initialize ChromaDB client (persistence handled automatically)
persist_directory = os.path.join(VECTOR_DB_DIR, "chroma_db")
client = chromadb.PersistentClient(path=persist_directory)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def get_directory_hash(directory):
    """Calculates a hash of the directory's contents."""
    hasher = hashlib.md5()
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'rb') as f:
                    while chunk := f.read(4096):
                        hasher.update(chunk)
            except Exception as e:
                print(f"Error hashing {file_path}: {e}")
    return hasher.hexdigest()

def load_and_embed_pdf(pdf_path, collection):
    """Loads, chunks, and embeds PDFs from a directory and its subdirectories, checking for changes."""

    file_hash = get_directory_hash(pdf_path)

    # Check if the PDF (by hash) is already in the collection
    results = collection.get(where={"source": pdf_path}, include=["metadatas"])
    if results['metadatas']:
        existing_metadata = results['metadatas'][0]
        if existing_metadata.get("file_hash") == file_hash:
            print(f"{pdf_path} (unchanged) already in the collection.")
            return
        
    if pdf_path.lower().endswith(".pdf"):
        groups = ", ".join(pdf_path.split("\\")[2:-1])
        if groups == "":
            groups = "all"
        _process_single_pdf(pdf_path, collection, os.path.basename(pdf_path), os.path.dirname(pdf_path), groups) # handle single pdf
    else:
        print(f"Error: {pdf_path} is not a valid PDF file or directory.")

def _process_single_pdf(pdf_path, collection, relative_path, base_path, groups = "all"):
    """Processes a single PDF file, checking for changes and embedding it."""

    file_hash = get_directory_hash(pdf_path)

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
        metadatas.append({"source": pdf_path, "chunk": i, "page": page_number, "file_hash": file_hash, "relative_path": relative_path, "groups": groups}) # include relative path
        ids.append(chunk_id)

    # If the PDF existed before, delete old entries first (to update changed content).
    if results['metadatas']:
        collection.delete(ids=results['ids'])  # Delete by IDs

    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    print(f"Added/Updated {len(chunks)} chunks from {pdf_path} to collection.")

def rebuild_database(pdf_directory, collection_name, collection):
    """Rebuilds the database if needed due to file changes."""
    files_in_db = set(meta['source'] for meta in collection.get(include=["metadatas"])['metadatas']) if collection.count()>0 else set()
    files_on_disk = set()

    # Listing all the PDF files.
    for root, dirs, files in os.walk(pdf_directory):
        for file in files:
            if file.endswith('.pdf'):
                files_on_disk.add(os.path.join(root, file))

    if files_in_db != files_on_disk:
        print("File changes detected. Rebuilding database...")
        client.delete_collection(name=collection_name) # Delete the collection
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        ) # Recreate it
        for pdf_path in files_on_disk:
            load_and_embed_pdf(pdf_path, collection)
    else:
        print("No file changes detected.")

def search_chroma(query, collection_name="pdf_collection", n_results=2, filter = "all"):
    """Searches ChromaDB for the given query."""
    collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={'groups': {'$in': [filter, "all"]}}
    )
    return results

