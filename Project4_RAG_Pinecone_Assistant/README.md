# 📄 Query Documents with Pinecone

A Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **Pinecone**, **Sentence Transformers**, and **Groq Llama**.

The application allows users to upload PDF documents, store embeddings in Pinecone, and ask questions based only on the uploaded documents.

---

## 🚀 Features

- 📄 Upload PDF documents
- 🔍 Semantic search using Sentence Transformers
- 🗂 Store embeddings in Pinecone
- 🤖 Answer questions using Groq Llama
- 💬 Streaming AI responses
- 📑 Displays source document names
- ⚡ Fast Retrieval-Augmented Generation (RAG)

---

## 🛠 Tech Stack

- Streamlit
- Pinecone
- Sentence Transformers
- Groq API
- PyPDF
- PyTorch

---

## 📂 Project Structure

```
Project4_Query_Documents/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/Kamal747/GenAIProjects.git
```

Move into the project folder

```bash
cd Project4_Query_Documents
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
python -m pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🔑 API Keys Required

### Groq API

https://console.groq.com/keys

### Pinecone API

https://app.pinecone.io/

---

## 📖 Workflow

1. Enter Groq API Key
2. Enter Pinecone API Key
3. Click **Initialize**
4. Upload a PDF
5. Document is converted into embeddings
6. Embeddings are stored in Pinecone
7. Ask questions about the document
8. AI retrieves relevant chunks
9. Groq Llama generates the answer
10. Source document names are displayed

---

## 📸 Demo

Upload a PDF

↓

Convert into chunks

↓

Generate embeddings

↓

Store in Pinecone

↓

Semantic Search

↓

Groq Llama

↓

Answer with Sources

---

## 🎯 Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Pinecone Vector Database
- Streamlit Application Development
- Groq LLM Integration
- PDF Processing
- Prompt Engineering

---

## 👨‍💻 Author

**Kamaleshwar S**

GitHub:
https://github.com/Kamal747

LinkedIn:
https://www.linkedin.com/in/kamaleshwar2804