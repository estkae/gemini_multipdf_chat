# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands

### Python Setup
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Google API key
# GOOGLE_API_KEY=your_actual_api_key_here
```

### Running the Application
```bash
# Run the Streamlit app
streamlit run app.py

# The app will be available at http://localhost:8501
```

### Docker Commands
```bash
# Build and run with Docker Compose
docker compose up --build

# Build for different architectures (e.g., for cloud deployment)
docker build --platform=linux/amd64 -t gemini-pdf-chatbot .

# Run Docker container
docker run -p 8501:8501 --env-file .env gemini-pdf-chatbot
```

### Common Development Tasks
```bash
# Check Python syntax issues
python -m py_compile app.py

# Format Python code (if black is installed)
pip install black
black app.py

# Type checking (if mypy is installed)
pip install mypy
mypy app.py
```

## Code Architecture

### Application Flow
1. **PDF Upload & Processing** (`get_pdf_text`, `get_text_chunks`):
   - Users upload PDFs via Streamlit sidebar
   - PyPDF2 extracts text from all pages
   - Text is split into 10,000 character chunks with 1,000 character overlap

2. **Vector Storage** (`get_vector_store`):
   - Google's embedding-001 model creates embeddings for each chunk
   - FAISS stores embeddings locally in `faiss_index/` directory
   - Embeddings enable semantic search across PDF content

3. **Question Answering** (`user_input`, `get_conversational_chain`):
   - User questions trigger similarity search in FAISS
   - Retrieved chunks provide context to Gemini Pro model
   - Custom prompt template ensures answers stay within PDF context
   - Chat history maintained in Streamlit session state

### Key Components
- **app.py**: Single-file application containing all logic
- **LangChain Integration**: Handles LLM chains, embeddings, and vector stores
- **Streamlit UI**: Chat interface with sidebar for PDF uploads
- **FAISS Vector DB**: Persistent storage for document embeddings

### Important Considerations
- The app requires `allow_dangerous_deserialization=True` for FAISS.load_local() due to security updates
- No built-in tests - manual testing required for changes
- Session state manages chat history - cleared on "Clear Chat History" button
- Vector store persists between sessions in `faiss_index/` directory

### Dependencies
- **streamlit**: Web UI framework
- **google-generativeai**: Google's Gemini API client
- **langchain**: LLM application framework
- **langchain_google_genai**: LangChain's Google integration
- **PyPDF2**: PDF text extraction
- **faiss-cpu**: Vector similarity search
- **chromadb**: Alternative vector DB (imported but not used)
- **python-dotenv**: Environment variable management