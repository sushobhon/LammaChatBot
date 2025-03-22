# Importing Necessary Libraries
import re
import json
import os
import chromadb
import Helper_functions as hf
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from chromadb.utils import embedding_functions


## Part 1:
# Specifying the model
# model = OllamaLLM(model= "deepseek-r1:8b", temperature= 0)
model = OllamaLLM(model= 'llama3.2:latest', temperature= 0)

template = """
You are a helpful Q&A assistant. Your goal is to answer user questions based on the provided context and conversation history. If the answer cannot be found within the provided information, respond with "I don't know."

Context:
{context}

Conversation History:
{conversation_history}

User Question:
{question}

**Answer:**
"""
# Also return the Source and Chunk number of the context.
# Creating The Prompt
prompt = ChatPromptTemplate.from_template(template= template)

# Creating a conversation chain
chain = prompt | model

## Part 2: 
# Defining Vector database name and collection name
VECTOR_DB_DIR = "VectorDatabase"
COLLECTION_NAME = "pdf_collection"

# Spefifying PDF Directory
pdf_directory = ".\\data"  # Current directory
os.makedirs(VECTOR_DB_DIR, exist_ok=True) # Create VectorDatabase directory if not present

# Initialize ChromaDB client (persistence handled automatically)
persist_directory = os.path.join(VECTOR_DB_DIR, "chroma_db")
client = chromadb.PersistentClient(path=persist_directory)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Get or create the collection:
if COLLECTION_NAME not in client.list_collections():
    print(f"Creating collection '{COLLECTION_NAME}'...")
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
else:
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)

# Loading or Building the Vector Database
hf.rebuild_database(      # hf.
    pdf_directory,
    collection_name= COLLECTION_NAME,
    collection= collection
    ) 

# Defining a function to handel the conversation history
def handle_conversation(filename="chat_history.json"):
    """Handles the conversation, storing only the last 5 exchanges."""

    try:
        with open(filename, 'r') as f:
            chat_history = json.load(f)
    except FileNotFoundError:
        chat_history = []

    filter = input("Enter the Group Name: ")

    print("BOT: Type your question to start. Type 'exit' to Exit chat!!!")

    while True:
        user_input = input("YOU: ")
        if user_input.lower() == 'exit':
            break
        
        # fininding Relevent Chunks of Data
        search_results = hf.search_chroma(user_input, filter= filter)

        # Creating context for LLM
        context = ""
        for i, result in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0])):
            text_chunk = result[0]
            metadata = result[1]
            context += f"Result: {i+1}\n"
            context += f"Source: {metadata['source']}, Chunk: {metadata['chunk']}\n"
            context += f"Text: {text_chunk}\n"

        # Convert chat_history to string for LLM input (if needed)
        conversation_string = ""
        for turn in chat_history:
            conversation_string += f"YOU: {turn['user']}\nBOT: {turn['bot']}\n"


        result = chain.invoke({
            "conversation_history": conversation_string,
            "context": context,
            "question": user_input
        }) #Pass the string here

        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL).strip()
        print(result)
        
        if result != "I don't know.":
            for entry in search_results["metadatas"][0]:
                source = entry['source']
                page = entry['page']
                chunk = entry['chunk']
                print(f"Source: {source}, Page: {page}, Chunk: {chunk}")


        # Update chat history (list of dictionaries)
        chat_history.append({"user": user_input, "bot": result})

        if len(chat_history) > 5:
            chat_history = chat_history[-5:]

        # Store the updated history
        with open(filename, 'w') as f:
            json.dump(chat_history, f, indent=4)

if __name__ == "__main__":
    handle_conversation()
