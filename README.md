# RAG QnA System

This repository contains the code for a Retrieval-Augmented Generation (RAG) Question Answering system. It allows users to ask questions based on provided context documents and previous conversation history.

## Features

* **Store Information:** Creates a Vector Database based on the pdf Documents and search for answering question.
* **Contextual Question Answering:** Search Answer in the database and pull out answer. 
* **Conversation History:** Maintains and utilizes conversation history for context.
* **"I don't know" Handling:** Gracefully handles questions that cannot be answered from the provided information.

## Access to Particular File:
All PDF file must be inside `data` folder. Any file in `data` folder can be accessed by all users. We can create multiple sub-folders inside `data` folder. The name of the sub-folder is the group name. Any file inside a particular sub-folder can be accessed by the members of that group. We can create multiple sub-folder inside any sub-folder.

# How to Run?

1. Install Ollama and download any model.
2. Clone the repository.
3. Create an environment based on requirment file.
4. Run run.py to chat.
5. To Add or Remove any document just add the document in 'data' folder.
