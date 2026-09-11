# MedNova
# MedNova 🩺

MedNova is a medical question-answering chatbot built with **Retrieval-Augmented Generation (RAG)**. It answers medical questions using content extracted from *The Gale Encyclopedia of Medicine*, combining semantic search over a FAISS vector store with a Groq-hosted LLM, and serves the chatbot through a simple Flask web interface.

## How It Works

1. **Ingestion** – The Gale Encyclopedia PDF is loaded and split into overlapping text chunks.
2. **Embedding** – Each chunk is embedded using a HuggingFace sentence-transformer model.
3. **Vector Store** – Embeddings are indexed and stored locally with FAISS.
4. **Retrieval** – On each user query, the most relevant chunk(s) are retrieved from FAISS.
5. **Generation** – The retrieved context and question are passed to a Groq LLM via a custom prompt, and a concise answer (2–3 lines) is generated.
6. **Web UI** – A Flask app exposes a chat-style interface backed by session-stored conversation history.

## Tech Stack

- **Python**
- **Flask** – web application and routing
- **LangChain** – RAG orchestration (`langchain`, `langchain-community`)
- **Groq** (`langchain_groq`) – LLM inference (`openai/gpt-oss-120b`)
- **HuggingFace** (`langchain_huggingface`) – text embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **FAISS** (`faiss-cpu`) – vector similarity search
- **PyPDF** – PDF parsing
- **python-dotenv** – environment variable management

## Project Structure

```
MedNova/
├── app/
│   ├── application.py          # Flask app & chat routes
│   ├── common/
│   │   ├── logger.py            # Logging setup
│   │   └── custom_exception.py  # Custom exception handling
│   ├── components/
│   │   ├── load_pdf.py          # PDF loading & chunking
│   │   ├── embeddings.py        # HuggingFace embedding model
│   │   ├── vector_store.py      # FAISS save/load logic
│   │   ├── retriever.py         # QA chain (retriever + LLM + prompt)
│   │   ├── llm.py               # Groq LLM loader
│   │   └── data_loader.py       # Orchestrates PDF → chunks → vectorstore
│   ├── config/
│   │   └── config.py            # Paths, chunk size, API key config
│   └── templates/
│       └── index.html           # Chat UI
├── data/                        # Source PDF(s) (e.g. Gale Encyclopedia of Medicine)
├── vectorstore/db_faiss/        # Generated FAISS index (created after ingestion)
├── logs/                        # Runtime logs
├── req.txt                      # Python dependencies
└── setup.py                     # Package setup
```

## Getting Started

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone https://github.com/Laxmipriya-Swain/MedNova.git
cd MedNova
pip install -r req.txt
```

### Configuration

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Build the Vector Store

Place your source PDF(s) in the `data/` folder, then run:

```bash
python -m app.components.data_loader
```

This loads the PDF(s), splits them into chunks, generates embeddings, and saves the FAISS index to `vectorstore/db_faiss`.

### Run the App

```bash
python -m app.application
```

The chatbot will be available at **http://localhost:5000**.

## Usage

Type a medical question into the chat box, and MedNova retrieves the most relevant passage from the encyclopedia and generates a concise, context-grounded answer. Use the **Clear** option to reset the conversation.

## ⚠️ Disclaimer

MedNova is built for educational and informational purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.

## Author

**Laxmipriya Swain**
[GitHub](https://github.com/Laxmipriya-Swain) · [LinkedIn](https://linkedin.com/in/laxmipriya-swain/)
