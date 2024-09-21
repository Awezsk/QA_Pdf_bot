# Interactive QA Bot

This project implements an interactive Question Answering (QA) bot using Streamlit, Pinecone, Cohere, and SentenceTransformer. The bot allows users to upload PDF documents, indexes their content, and then answers questions based on the document's contents.

## Features

- PDF document upload and processing
- Document indexing using Pinecone vector database
- Question answering using Cohere's language model
- Interactive web interface built with Streamlit

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7+
- Pinecone API key
- Cohere API key

## Installation

1. Clone this repository:
   ```
   git clone git clone https://github.com/Awezsk/QA_Pdf_bot.git
   cd interactive-qa-bot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.streamlit/secrets.toml` file in the project directory and add your API keys:
   ```toml
   PINECONE_API_KEY = "your_pinecone_api_key"
   COHERE_API_KEY = "your_cohere_api_key"
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload a PDF document using the file uploader.

4. Wait for the document to be processed and indexed.

5. Ask questions about the document in the text input field.

6. View the generated answers and relevant document segments.

## How it Works

1. **Document Processing**: The app uses PyPDF2 to extract text from uploaded PDF files and splits it into chunks.

2. **Indexing**: Document chunks are embedded using SentenceTransformer and indexed in Pinecone.

3. **Question Answering**:
   - The user's question is embedded and used to retrieve relevant chunks from Pinecone.
   - Relevant chunks are combined with the question to form a prompt for Cohere's language model.
   - The language model generates an answer based on the provided context.

4. **Result Display**: The app shows the generated answer along with the relevant document segments.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
