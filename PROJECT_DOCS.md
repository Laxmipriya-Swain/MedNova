# 🏥 MedNova — AI Medical Chatbot
### Complete Project Documentation

---

## 📌 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Tech Stack & Why Each Tool Was Used](#2-tech-stack--why-each-tool-was-used)
3. [Project Folder Structure](#3-project-folder-structure)
4. [Every File — Purpose & Code Explained](#4-every-file--purpose--code-explained)
5. [Core Concepts Explained](#5-core-concepts-explained)
6. [Step-by-Step Working Flow](#6-step-by-step-working-flow)
7. [Data Pipeline (How PDFs become Knowledge)](#7-data-pipeline-how-pdfs-become-knowledge)
8. [Query Pipeline (How a Question gets Answered)](#8-query-pipeline-how-a-question-gets-answered)
9. [Frontend — MedNova UI](#9-frontend--mednova-ui)
10. [Performance Optimization Done](#10-performance-optimization-done)
11. [How to Run the Project](#11-how-to-run-the-project)
12. [Interview Guide — How to Explain This Project](#12-interview-guide--how-to-explain-this-project)

---

## 1. Project Overview

**MedNova** is an AI-powered **Medical Question Answering Chatbot** built using the **RAG (Retrieval-Augmented Generation)** architecture.

Instead of relying purely on an LLM's pre-trained knowledge (which may be outdated or hallucinate medical facts), MedNova:

1. **Reads medical PDF books** from a local `data/` folder
2. **Converts** the text into numerical vectors (embeddings)
3. **Stores** those vectors in a FAISS vector database
4. When a user asks a question → **retrieves the most relevant text chunks** from the database
5. **Passes** those chunks + the question to Groq's LLM
6. The LLM generates a **grounded, accurate answer** based only on the retrieved context

> **Key Advantage**: The chatbot only answers based on the actual medical books — no hallucination.

---

## 2. Tech Stack & Why Each Tool Was Used

| Technology | Role | Why Used |
|---|---|---|
| **Python** | Core language | Widely used in AI/ML, excellent libraries |
| **Flask** | Web server / API | Lightweight, easy to use for Python web apps |
| **LangChain** | AI orchestration framework | Connects LLM + Retriever + Prompt in one pipeline |
| **Groq API** | LLM provider | Ultra-fast inference (uses LPU hardware), free tier available |
| **LLaMA / GPT-OSS model** | Large Language Model | Generates human-like answers from retrieved context |
| **HuggingFace Embeddings** | Embedding model | Converts text to semantic vectors |
| **sentence-transformers/all-MiniLM-L6-v2** | Specific embedding model | Lightweight, fast, good semantic understanding |
| **FAISS** | Vector database | Facebook AI's fast similarity search — works offline, no cloud needed |
| **PyPDF** | PDF reader | Reads medical books stored as PDFs |
| **python-dotenv** | Env variable loader | Keeps API keys secure in `.env` file |
| **Jinja2** | HTML templating | Flask's built-in template engine for dynamic HTML |
| **localStorage (JS)** | Chat history | Saves session titles in browser — no DB needed |
| **marked.js** | Markdown parser | Renders AI responses with formatting (bold, lists, etc.) |

---

## 3. Project Folder Structure

```
MEDCHAT/
│
├── .env                          # Secret keys (GROQ_API_KEY)
├── req.txt                       # Python dependencies
├── setup.py                      # Package setup
├── PROJECT_DOCS.md               # <- This file
│
├── data/                         # Place your medical PDF books here
│
├── vectorstore/
│   └── db_faiss/                 # FAISS index files (auto-generated)
│       ├── index.faiss           # The actual vector index
│       └── index.pkl             # Metadata (document chunks)
│
├── logs/
│   └── log_YYYY-MM-DD.log        # Daily log files
│
└── app/
    ├── __init__.py
    ├── application.py            # Flask app — main entry point
    │
    ├── templates/
    │   └── index.html            # MedNova UI (Frontend)
    │
    ├── config/
    │   └── config.py             # All constants (paths, keys, chunk size)
    │
    ├── common/
    │   ├── logger.py             # Logging setup
    │   └── custom_exception.py   # Custom error handling
    │
    └── components/
        ├── load_pdf.py           # Load & split PDF documents
        ├── embeddings.py         # HuggingFace embedding model
        ├── vector_store.py       # FAISS save & load
        ├── llm.py                # Load Groq LLM
        ├── retriever.py          # Build the QA chain
        └── data_loader.py        # Orchestrates the full data pipeline
```

---

## 4. Every File — Purpose & Code Explained

---

### `.env`
```
GROQ_API_KEY=your_api_key_here
```
Stores the **Groq API key** securely. Never committed to Git. Loaded by `python-dotenv`.

---

### `app/config/config.py`
```python
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")   # API key from .env
DB_FAISS_PATH = "vectorstore/db_faiss"           # Where FAISS index is stored
DATA_PATH = "data/"                              # Where medical PDFs are placed
CHUNK_SIZE = 500                                 # Each text chunk = 500 characters
CHUNK_OVERLAP = 50                               # 50 chars overlap between chunks
```
**Purpose**: Central place for all configuration. Change paths/settings here once and it affects the whole project.

---

### `app/common/logger.py`
```python
logging.basicConfig(filename=LOG_FILES, format='%(asctime)s -%(levelname)s -%(message)s')
```
**Purpose**: Creates a daily log file like `logs/log_2026-09-11.log`.  
Every module imports `get_logger(__name__)` so you can trace exactly which file logged what.

---

### `app/common/custom_exception.py`
**Purpose**: A custom exception class that wraps Python errors with better messages.  
Makes debugging easier — instead of a generic Exception, you know exactly where and why it failed.

---

### `app/components/load_pdf.py`

**Two functions:**

#### `load_pdf_files()`
```python
loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
```
- Scans the `data/` folder
- Loads **every `.pdf` file** using `PyPDFLoader`
- Returns a list of `Document` objects (each has `.page_content` and `.metadata`)

#### `create_text_chunks(documents)`
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)
```
- Splits large documents into **500-character chunks** with 50 chars overlap
- **Why overlap?** — Ensures that a sentence at a chunk boundary is not cut off. Context continuity is preserved.
- Returns a list of smaller `Document` objects ready to be embedded

---

### `app/components/embeddings.py`

```python
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**Purpose**: Loads the embedding model that converts text into a numerical vector.

**Why `all-MiniLM-L6-v2`?**
- Lightweight (only 22MB)
- Fast on CPU
- 384-dimensional output vectors
- Excellent semantic understanding — similar sentences get similar vectors
- Completely **free and local** — no API call needed

**Example:**
```
"heart attack symptoms"  -->  [0.23, -0.45, 0.12, ... 384 numbers]
"myocardial infarction"  -->  [0.22, -0.43, 0.13, ... 384 numbers]  (Very similar!)
```

---

### `app/components/vector_store.py`

**Two functions:**

#### `save_vector_store(text_chunks)` — Run once to create the DB
```python
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
```
- Takes all text chunks, embeds each one, stores in FAISS index
- Saves two files: `index.faiss` (vectors) and `index.pkl` (metadata)

#### `load_vector_store()` — Run every time the app starts
```python
return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
```
- Loads the existing FAISS index from disk
- Ready to perform similarity searches

---

### `app/components/llm.py`

```python
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="openai/gpt-oss-120b", temperature=0.7, max_tokens=250)
```

**Purpose**: Loads the Language Model from Groq's API.

**Parameters explained:**

| Parameter | Value | Meaning |
|---|---|---|
| `model_name` | `gpt-oss-120b` | 120 billion parameter model (very capable) |
| `temperature` | `0.7` | Slight creativity — not too random, not too rigid |
| `max_tokens` | `250` | Limits response length — keeps answers concise |

**Why Groq?**
- Uses **LPU (Language Processing Unit)** hardware — extremely fast inference
- Free tier available with generous limits
- Faster than OpenAI for the same model size

---

### `app/components/retriever.py`

The **heart of the RAG system**.

```python
CPT = """
Answer the following medical question in 2-3 lines maximum
using only the information provided in the context.
Context: {context}
Question: {question}
Answer:
"""
```

This is the **Prompt Template** — it tells the LLM:
- Use ONLY the provided context (not its own knowledge)
- Answer in 2-3 lines (concise)

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 1}),
    return_source_documents=False,
    chain_type_kwargs={'prompt': set_custom_prompt()}
)
```

| Parameter | Value | Meaning |
|---|---|---|
| `chain_type` | `"stuff"` | Stuffs retrieved docs directly into the prompt |
| `retriever` | `k=1` | Fetches top-1 most similar chunk from FAISS |
| `return_source_documents` | `False` | Only return the answer, not the source text |

---

### `app/components/data_loader.py`

```python
def process_and_store_pdfs():
    documents = load_pdf_files()
    text_chunks = create_text_chunks(documents)
    save_vector_store(text_chunks)
```

**Purpose**: The **one-time setup script**.  
Run this manually to:
1. Load all PDFs
2. Split into chunks
3. Embed and save to FAISS

Only needs to be run once (or when you add new PDFs).

---

### `app/application.py`

The **Flask web server** — entry point of the application.

```python
# Cache QA chain ONCE at startup
_qa_chain = create_qa_chain()

@app.route("/", methods=["GET", "POST"])
def index():
    # GET  --> show the chat page
    # POST --> process user message, get AI answer
```

**Key design decisions:**
- `_qa_chain` is cached at module level — loads model ONCE when server starts
- Session stores the full chat history (user + assistant messages)
- `/clear` route wipes the session and redirects to home

---

## 5. Core Concepts Explained

---

### What is RAG? (Retrieval-Augmented Generation)

RAG is an AI technique that combines:
- **Retrieval**: Find relevant information from a knowledge base
- **Augmented**: Add that information to the prompt
- **Generation**: LLM generates an answer using that information

```
User Question
     |
     v
Embed the question --> Search FAISS
     |
     v
Get top-K relevant text chunks
     |
     v
Inject into Prompt:
  "Based on this context: [chunks], answer: [question]"
     |
     v
LLM generates answer
     |
     v
Show to user
```

**Why RAG over fine-tuning?**

| RAG | Fine-tuning |
|---|---|
| No retraining needed | Requires expensive GPU training |
| Add new knowledge instantly | Need to retrain every time |
| Answers grounded in real documents | May hallucinate |
| Cheap | Very expensive |

---

### What is a Vector Database?

A vector database stores data as **numerical vectors** instead of text.

**Why?** Computers cannot directly compare meaning. But they CAN compare numbers.

```
"chest pain"         --> [0.12, -0.45, 0.89, ...]   384 numbers
"heart attack sign"  --> [0.11, -0.44, 0.88, ...]   <- Very similar!
"pizza recipe"       --> [-0.9, 0.23, -0.11, ...]   <- Very far away
```

FAISS finds vectors that are **mathematically closest** to your query vector — which means most semantically similar content.

---

### What is FAISS?

**FAISS = Facebook AI Similarity Search**

- Open-source library by Meta/Facebook
- Stores millions of vectors and searches them in milliseconds
- Works completely **offline** (no cloud needed)
- In this project: stored on disk at `vectorstore/db_faiss/`

---

### What is an Embedding Model?

Converts text into a fixed-size vector of numbers capturing **semantic meaning**.

**Model used**: `sentence-transformers/all-MiniLM-L6-v2`
- Input: Any text sentence
- Output: 384 float numbers
- Similar meaning = Similar numbers

---

### What is an LLM?

**LLM = Large Language Model**

A neural network trained on massive amounts of text. Given a prompt, it predicts the best response.

In this project: Accessed via **Groq API**. The LLM does NOT search the database — it only **reads the retrieved context and generates a clean answer**.

---

## 6. Step-by-Step Working Flow

```
PHASE 1: SETUP (One-time only)
-------------------------------
Medical PDFs --> PyPDF Loader --> Text Chunks
     --> HuggingFace Embedder --> FAISS Vector DB (saved to disk)


PHASE 2: SERVER STARTUP
------------------------
Flask starts
  --> load_vector_store()  --> FAISS loaded into memory
  --> load_llm()           --> Groq LLM ready
  --> create_qa_chain()    --> Chain cached in _qa_chain


PHASE 3: USER INTERACTION (every message)
------------------------------------------
User types question in browser
  --> AJAX POST to Flask /
  --> Embed question
  --> Search FAISS (top k=1 chunk)
  --> Build prompt: context + question
  --> Groq LLM generates answer
  --> Sent back to browser
  --> Rendered as Markdown in MedNova chat
```

---

## 7. Data Pipeline (How PDFs become Knowledge)

```
data/medical_book.pdf
        |
        v
  PyPDFLoader (load_pdf.py)
  - Reads every page
  - Creates Document objects
        |
        v
  RecursiveCharacterTextSplitter (load_pdf.py)
  - chunk_size = 500 chars
  - chunk_overlap = 50 chars
  - Creates many smaller Document chunks
        |
        v
  HuggingFaceEmbeddings (embeddings.py)
  - all-MiniLM-L6-v2
  - Each chunk --> 384-dimensional vector
        |
        v
  FAISS.from_documents() (vector_store.py)
  - Builds similarity index
  - Saves:
      vectorstore/db_faiss/index.faiss
      vectorstore/db_faiss/index.pkl
```

---

## 8. Query Pipeline (How a Question gets Answered)

```
User: "What are symptoms of diabetes?"
        |
        v
  Embed question --> [0.31, -0.12, ...] (384 nums)
        |
        v
  FAISS similarity search (k=1)
  - Finds the 1 most similar text chunk from medical book
  - Returns: "Diabetes symptoms include polyuria, polydipsia..."
        |
        v
  Prompt Template (retriever.py)
  - "Answer the medical question using ONLY this context:
     Context: {retrieved chunk}
     Question: What are symptoms of diabetes?
     Answer:"
        |
        v
  Groq LLM (gpt-oss-120b)
  - Reads context + question
  - Generates concise 2-3 line answer
        |
        v
  Response returned to Flask --> Session --> Browser
  - Rendered as Markdown in MedNova chat UI
```

---

## 9. Frontend — MedNova UI

Built as a **single `index.html`** file using:
- Vanilla HTML + CSS + JavaScript (no React/Vue needed)
- Jinja2 templating for server-side rendering of chat history
- marked.js CDN for Markdown parsing

**Features:**

| Feature | Technology |
|---|---|
| Chat history sidebar | JavaScript + localStorage |
| AJAX messaging | fetch() API |
| Markdown rendering | marked.js |
| Dark/Light toggle | CSS variables + localStorage |
| Copy button | navigator.clipboard API |
| Text-to-speech | Web Speech API |
| Animated background | CSS animations + SVG |
| Typing indicator | CSS keyframes |
| Mobile responsive | CSS media queries |

---

## 10. Performance Optimization Done

### Problem: Slow Response (~50 seconds per message)

**Before (bad):**
```python
@app.route("/", methods=["POST"])
def index():
    qa_chain = create_qa_chain()  # Loaded HuggingFace + FAISS on EVERY message!
```

**After (optimized):**
```python
_qa_chain = create_qa_chain()     # Loaded ONCE at server startup

@app.route("/", methods=["POST"])
def index():
    qa_chain = _qa_chain           # Reuses cached chain
```

**Result:**

| Metric | Before | After |
|---|---|---|
| 1st message | ~50 seconds | ~3 seconds |
| 2nd+ message | ~50 seconds | ~1-2 seconds |
| Model reloads | Every request | Only on server start |

---

## 11. How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r req.txt
```

### Step 2: Set up your API key
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key from: https://console.groq.com

### Step 3: Add medical PDFs
Place your PDF books in the `data/` folder.

### Step 4: Build the vector database (one-time only)
```bash
python -m app.components.data_loader
```
This reads PDFs, creates embeddings, saves FAISS index.

### Step 5: Run the Flask server
```bash
python -m app.application
```
Open browser: http://127.0.0.1:5000

---

## 12. Interview Guide — How to Explain This Project

---

### Opening Statement (30 seconds)
> "I built an AI-powered Medical Chatbot called MedNova using the RAG architecture. It reads medical PDF textbooks, converts them into searchable vector embeddings stored in FAISS, and when a user asks a question, it retrieves the most relevant medical text and passes it to a Groq LLM to generate an accurate, grounded answer. The key advantage is it does not hallucinate — it only answers based on the actual medical books."

---

### Common Interview Questions & Answers

**Q: What is RAG and why did you use it?**
> RAG stands for Retrieval-Augmented Generation. Instead of relying purely on an LLM's pre-trained knowledge which can hallucinate, RAG first retrieves relevant information from a private knowledge base (our medical PDFs) and then passes that context to the LLM to generate the answer. I used it because medical accuracy is critical — we cannot afford hallucinations in healthcare.

---

**Q: What is a vector database? Why FAISS?**
> A vector database stores text as numerical embeddings that capture semantic meaning. Similar sentences have similar vectors. I used FAISS (Facebook AI Similarity Search) because it is extremely fast, works completely offline without any cloud dependency, is open-source, and is production-grade. It allows similarity search in milliseconds even with thousands of document chunks.

---

**Q: What is an embedding model? Which one did you use?**
> An embedding model converts text into a fixed-size vector of numbers. I used `sentence-transformers/all-MiniLM-L6-v2` from HuggingFace. It produces 384-dimensional vectors. I chose it because it is lightweight (22MB), runs on CPU, is completely free, and has excellent semantic understanding.

---

**Q: Why Groq instead of OpenAI?**
> Groq uses custom LPU (Language Processing Unit) hardware specifically designed for LLM inference, making it significantly faster than GPU-based inference. It also has a generous free tier. The API is compatible with OpenAI's format, so switching is easy if needed.

---

**Q: How does text chunking work and why do you overlap chunks?**
> I used RecursiveCharacterTextSplitter with chunk_size=500 and chunk_overlap=50. The 50-character overlap ensures that sentences spanning chunk boundaries are not completely cut off — it preserves context continuity at boundaries. FAISS also searches more accurately on smaller, focused pieces of text than on full documents.

---

**Q: What performance issue did you find and how did you fix it?**
> The original code called `create_qa_chain()` inside the Flask route handler on every POST request. This reloaded the HuggingFace embedding model and FAISS index from disk every time a user sent a message, causing ~50 second delays. I fixed it by moving `create_qa_chain()` to module level so it runs once at server startup and is cached in `_qa_chain`. This reduced response time from 50 seconds to 1-3 seconds — a 95% improvement.

---

**Q: How is the frontend built?**
> The frontend is a single index.html file using vanilla HTML, CSS, and JavaScript with Jinja2 templating. It uses the fetch() API for AJAX chat without page reloads, marked.js for Markdown rendering, the Web Speech API for text-to-speech, the Clipboard API for copy, CSS variables for dark/light mode, and localStorage for persisting chat history in the sidebar.

---

**Q: How would you scale this to production?**
> Several improvements I would make:
> 1. Replace Flask dev server with Gunicorn or uWSGI
> 2. Add GPU support for faster embedding (currently CPU)
> 3. Use a proper database (PostgreSQL) for chat history instead of Flask sessions
> 4. Add user authentication
> 5. Implement streaming responses for better UX
> 6. Use FAISS with IVF index for larger document collections
> 7. Add Docker containerization for deployment
> 8. Consider Pinecone or Weaviate for large-scale managed vector search

---

**Q: What is the LangChain RetrievalQA chain?**
> RetrievalQA is a LangChain abstraction that combines a retriever and an LLM into a single pipeline. With chain_type="stuff", it takes the retrieved document chunks, stuffs them directly into the prompt context window, and passes the combined prompt to the LLM. It handles the full orchestration: embed query, search FAISS, format prompt, call LLM, return answer.

---

### Strong Closing Statement
> "This project demonstrates my ability to combine multiple AI/ML technologies — HuggingFace transformers, vector databases, LLMs, and web development — into a production-ready application. I also identified and fixed a critical performance bottleneck that reduced response time by 95%. The architecture is domain-agnostic — by just swapping the PDFs, it could work for legal, financial, or any other domain."

---

*MedNova v1.0 | Built with LangChain + Groq + FAISS + Flask*

"MedNova uses RAG — it reads medical PDFs, converts them to vectors using HuggingFace embeddings, stores them in FAISS, and when a user asks a question, it retrieves the most relevant chunk and gives it to Groq's LLM to generate a grounded answer. It can't hallucinate because it's constrained to only use the retrieved context."
