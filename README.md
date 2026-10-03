\# 📚 RAG Document Q\&A



A Retrieval-Augmented Generation (RAG) application that allows users to

upload one or more PDF documents and ask questions about their content.



The application extracts text from PDFs, creates semantic embeddings,

retrieves relevant document chunks, and uses a locally running LLM through

Ollama to generate grounded answers.



\---



\## 🚀 Features



\- Upload one or more PDF documents

\- Extract text from PDF files

\- Split documents into searchable chunks

\- Generate semantic embeddings

\- Perform similarity-based retrieval

\- Rank documents based on relevance

\- Generate answers using a local LLM

\- Support conversational follow-up questions

\- Display source documents

\- Display page numbers

\- Display similarity scores

\- Prevent unsupported answers

\- Run locally without sending document content to a cloud LLM



\---



\## 🏗️ Architecture



```text

&#x20;               PDF Documents

&#x20;                     │

&#x20;                     ▼

&#x20;             PDF Text Extraction

&#x20;                     │

&#x20;                     ▼

&#x20;                  Chunking

&#x20;                     │

&#x20;                     ▼

&#x20;           Sentence Transformers

&#x20;                     │

&#x20;                     ▼

&#x20;             Vector Embeddings

&#x20;                     │

&#x20;                     ▼

&#x20;         Document Relevance Ranking

&#x20;                     │

&#x20;                     ▼

&#x20;           Relevant Chunk Retrieval

&#x20;                     │

&#x20;                     ▼

&#x20;                Ollama LLM

&#x20;                     │

&#x20;                     ▼

&#x20;            Grounded Answer

&#x20;                     │

&#x20;                     ▼

&#x20;           Answer + Source Pages



🛠️ Technologies

Python

Streamlit

PyPDF

NumPy

Sentence Transformers

Ollama

Llama 3.2 3B

Git / GitHub



📂 Project Structure

rag-document-qa/

│

├── app/

│   └── app.py

│

├── src/

│   ├── build\_vector\_store.py

│   ├── chunking.py

│   ├── document\_loader.py

│   ├── embeddings.py

│   ├── ollama\_test.py

│   ├── rag\_pipeline.py

│   ├── search.py

│   └── vector\_store.py

│

├── data/

│   └── vector\_store/

│       ├── chunks.json

│       └── embeddings.npy

│

├── .gitignore

├── README.md

└── requirements.txt



⚙️ Installation

1\. Clone the repository

git clone https://github.com/narigemanikanta/rag-document-qa.git

cd rag-document-qa

2\. Create a virtual environment



Windows:



python -m venv .venv



Activate it:



.venv\\Scripts\\Activate.ps1

3\. Install dependencies

pip install -r requirements.txt

4\. Install and prepare Ollama



Install Ollama and download the required model:



ollama pull llama3.2:3b



Make sure Ollama is running before starting the application.



▶️ Run the Application



From the project root:



streamlit run app/app.py



The Streamlit application will open in your browser.



💬 Example Questions

Student Handbook

How many ECTS credits are required?



Example answer:



Students should complete 120 ECTS credits for

the full master's program.

Cover Letter

What position is mentioned in the cover letter?



Example answer:



The position mentioned in the cover letter is

Working Student with the Mobile Radio Testers team.

Follow-up Question

What team is it for?



The application uses the conversation context to understand

the reference to the previously discussed position.



.



🛡️ Grounded Answers



The system is designed to answer questions using the uploaded

documents rather than relying on unsupported information.



For example, when asked:



What is the salary for the Working Student position?



if the salary is not present in the uploaded documents, the

application responds:



I could not find relevant information in the documents.



This provides a basic safeguard against unsupported or

hallucinated answers.



🔎 Retrieval Pipeline



The application performs the following process:



PDF documents are uploaded.

Text is extracted from the documents.

Extracted text is divided into chunks.

Chunks are converted into semantic embeddings.

The user question is converted into an embedding.

Documents are ranked based on semantic relevance.

Relevant chunks are retrieved.

Retrieved context is passed to the local Ollama model.

The model generates a grounded answer.

Retrieved sources and page numbers are displayed.

📊 Example RAG Workflow

User Question

&#x20;     │

&#x20;     ▼

Question Embedding

&#x20;     │

&#x20;     ▼

Document Ranking

&#x20;     │

&#x20;     ▼

Semantic Similarity Search

&#x20;     │

&#x20;     ▼

Relevant Chunks

&#x20;     │

&#x20;     ▼

Context Construction

&#x20;     │

&#x20;     ▼

Llama 3.2 3B

&#x20;     │

&#x20;     ▼

Grounded Answer

&#x20;     │

&#x20;     ▼

Sources + Page Numbers

🔐 Privacy



The project is designed around local document processing and

local LLM inference through Ollama.



Uploaded PDFs used during development are excluded from the

Git repository through .gitignore.



🧪 Testing



The application has been tested with multiple PDF documents

including:



An AI \& Data Science student handbook

A Working Student cover letter



Tested functionality includes:



PDF upload

Multi-document processing

Semantic retrieval

Document-aware retrieval

Question answering

Follow-up questions

Source attribution

Unsupported-question handling

🚧 Future Improvements



Potential improvements include:



Hybrid keyword + semantic search

Cross-encoder reranking

Improved chunking strategies

Better conversation-aware retrieval

RAG evaluation metrics

Automated evaluation datasets

Docker support

Cloud deployment

Authentication

Improved document filtering

👨‍💻 Author



Manikanta Narige



M.Sc. Artificial Intelligence \& Data Science



Germany



GitHub:



https://github.com/narigemanikanta







