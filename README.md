# Claude PDF Chatbot

Claude PDF Chatbot is a Streamlit-based application that allows users to chat with Claude Sonnet, Anthropic's conversational AI model, trained on PDF documents. The chatbot extracts information from uploaded PDF files and answers user questions based on the provided context.
<https://gmultichat.streamlit.app/>

<https://github.com/kaifcoder/gemini_multipdf_chat/assets/57701861/f6a841af-a92d-4e54-a4fd-4a52117e17f6>

## Features

- **Multiple Input Methods:** 
  - Upload multiple files directly (PDF, TXT, DOCX, DOC, PNG, JPG, JPEG, TIFF, BMP)
  - Specify a folder path to process all supported files in that directory
- **Multi-Format Support:** Supports PDF, TXT, DOCX, DOC, and image formats (PNG, JPG, JPEG, TIFF, BMP).
- **OCR Support:** 
  - Automatic OCR for scanned PDFs and images
  - Fallback to OCR when regular text extraction fails
  - Supports English and German text recognition
- **Text Extraction:** Extracts text from all supported document types with intelligent OCR fallback.
- **Conversational AI:** Uses Claude 3.5 Sonnet, Anthropic's advanced AI model, to answer user questions.
- **Chat Interface:** Provides a chat interface to interact with the chatbot.

## Getting Started

If you have docker installed, you can run the application using the following command:

- Obtain an Anthropic API key and set it in the `.env` file.

   ```.env
   ANTHROPIC_API_KEY=your_api_key_here
   ```

```bash
docker compose up --build
```

Your application will be available at <http://localhost:8501>.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References

- [Docker's Python guide](https://docs.docker.com/language/python/)

## Local Development

Follow these instructions to set up and run this project on your local machine.

   **Note:** This project requires Python 3.10 or higher.

### Prerequisites for OCR

The OCR functionality requires Tesseract to be installed on your system:

- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr tesseract-ocr-deu`
- **macOS:** `brew install tesseract`
- **Windows:** Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

For PDF to image conversion (required for OCR on scanned PDFs):
- **Ubuntu/Debian:** `sudo apt-get install poppler-utils`
- **macOS:** `brew install poppler`
- **Windows:** 
  1. Download the latest release from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
  2. Extract the archive to a folder (e.g., `C:\poppler`)
  3. Add `C:\poppler\Library\bin` to your system PATH environment variable
  4. Restart your terminal/command prompt
  - **Alternative for Conda users:** `conda install -c conda-forge poppler`

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/claude-pdf-chatbot.git
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Anthropic API Key:**
   - Obtain an Anthropic API key and set it in the `.env` file.

   ```bash
   ANTHROPIC_API_KEY=your_api_key_here
   ```

4. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

5. **Add Documents:**
   - **Option 1 - Upload Files:** Use the "Upload Files" tab in the sidebar to upload PDF, TXT, DOCX, or DOC files.
   - **Option 2 - Select Folder:** Use the "Select Folder" tab to specify a folder path containing your documents.
   - Click on "Process Uploaded Files" or "Process Folder" to extract text and generate embeddings.

6. **Chat Interface:**
   - Chat with the AI in the main interface.

## Project Structure

- `app.py`: Main application script.
- `.env`: file which will contain your environment variable.
- `requirements.txt`: Python packages required for working of the app.
- `README.md`: Project documentation.

## Dependencies

- PyPDF2
- langchain
- python-docx
- Streamlit
- anthropic
- sentence-transformers
- dotenv

## Acknowledgments

- [Anthropic Claude](https://www.anthropic.com/): For providing the Claude 3.5 Sonnet language model.
- [Streamlit](https://streamlit.io/): For the user interface framework.
