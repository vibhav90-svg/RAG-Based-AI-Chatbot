
# ✦ RAG-Based AI Chatbot

An intelligent AI-powered chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded PDF documents.

The system retrieves relevant information from documents using semantic and hybrid search and generates context-aware responses using NVIDIA Nemotron.

## 🚀 Features

- PDF Document Loading
- Text Chunking
- Text Embeddings using FastEmbed
- Vector Storage using Qdrant
- Semantic Search
- Hybrid Search
- AI-powered Question Answering
- Interactive Streamlit Interface
- NVIDIA Nemotron Integration

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core Development |
| Streamlit | User Interface |
| FastEmbed | Text Embeddings |
| Qdrant | Vector Database |
| NVIDIA API | AI Response Generation |
| PyPDF | PDF Processing |

## 🏗️ Project Architecture

PDF Documents → Text Extraction → Text Chunking → Embeddings → Qdrant Vector Database → Retrieval → NVIDIA Nemotron → AI Response

## 📂 Project Structure

```text
RAG/
│
├── Document_Loader/
│   ├── embeddings.py
│   ├── hybrid_search.py
│   ├── load_pdf.py
│   ├── nvidia_llm.py
│   ├── semantic_search.py
│   ├── text_chunking.py
│   └── vector_store.py
│
├── data/
├── src/
│   └── rag_project/
├── app.py
├── rag_chatbot.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Based-AI-Chatbot.git
```

### 2. Open Project Folder

```bash
cd RAG-Based-AI-Chatbot
```

### 3. Create Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate Environment (Windows)

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure NVIDIA API Key

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

Never upload your API key to GitHub.

### 7. Run the Application

```bash
streamlit run app.py
```

## 🧠 How It Works

1. PDF documents are loaded and their text is extracted.
2. Extracted text is divided into smaller chunks.
3. FastEmbed converts chunks into vector embeddings.
4. Qdrant stores the generated vectors.
5. User queries retrieve relevant document information.
6. NVIDIA Nemotron generates responses using the retrieved context.

## 🎯 Project Objective

The objective of this project is to develop an intelligent document-based question-answering system that provides relevant and context-aware answers using Retrieval-Augmented Generation.

## 🔮 Future Improvements

- Multiple PDF Upload Support
- Chat History Management
- Source Document References
- Improved Retrieval Accuracy
- Enhanced User Interface

## 👨‍💻 Developer

**Vibhav Pandey**

B.Tech – Computer Science Engineering (Data Science & AI)

GitHub: [@vibhav90-svg](https://github.com/vibhav90-svg)

---

⭐ If you find this project useful, consider giving it a star!
