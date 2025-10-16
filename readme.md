# 📚 Study Bot AI

An intelligent AI-powered study assistant that helps you search and extract information from your course materials. Upload your documents and ask questions to get contextual answers based on your study materials.

## Features

- **Multi-format Support**: Upload PDF, DOCX, or TXT files
- **Smart Text Extraction**: Automatically extracts and processes text from various document formats
- **Intelligent Search**: Leverages Azure Cognitive Search to find relevant content from your documents
- **AI-Powered Responses**: Uses Azure OpenAI to generate accurate answers based solely on your study materials
- **Secure Storage**: Documents are securely stored in Azure Blob Storage
- **Chunked Processing**: Large documents are automatically split into manageable chunks with overlap for better context

## Prerequisites

Before running Study Bot AI, ensure you have:

- Python 3.8 or higher
- Azure account with the following services set up:
  - Azure Cognitive Search
  - Azure OpenAI Service
  - Azure Blob Storage
- Access to the following credentials:
  - Azure Search endpoint and API key
  - Azure OpenAI endpoint and API key
  - Azure Blob Storage connection string

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd study-bot-ai
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export SEARCH_ENDPOINT="https://<your-search-service>.search.windows.net"
export SEARCH_KEY="<your-search-key>"
export AZURE_OPENAI_ENDPOINT="https://<your-openai-resource>.openai.azure.com/"
export AZURE_OPENAI_KEY="<your-openai-key>"
export BLOB_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=<your-account>;AccountKey=<your-key>;EndpointSuffix=core.windows.net"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-35-turbo"  # Optional, defaults to gpt-35-turbo
export AZURE_SEARCH_INDEX_NAME="collegeproject_index"  # Optional, defaults to collegeproject_index
```

## Usage

### Running Locally

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Running with Docker

Use the provided startup script:
```bash
bash startup.sh
```

The application will be accessible at `http://0.0.0.0:8000`

### Using the Application

1. **Upload Documents**: Click the file uploader in the sidebar and select your PDF, DOCX, or TXT files
2. **Wait for Processing**: The application will extract, chunk, and index your documents
3. **Ask Questions**: Type your question in the text input field
4. **Get Answers**: The AI will search your documents and provide an answer based only on the content you've uploaded

## How It Works

1. **Text Extraction**: Documents are processed to extract readable text content
2. **Chunking**: Large texts are split into 5000-character chunks with 1000-character overlap to maintain context
3. **Indexing**: Content chunks are uploaded to Azure Cognitive Search for fast retrieval
4. **Search**: When you ask a question, the app searches the index for relevant content
5. **AI Response**: Azure OpenAI generates an answer based only on the retrieved documents
6. **Storage**: Original documents are also stored in Azure Blob Storage for reference

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SEARCH_ENDPOINT` | Azure Search service endpoint | Required |
| `SEARCH_KEY` | Azure Search API key | Required |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint | Required |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key | Required |
| `BLOB_CONNECTION_STRING` | Azure Blob Storage connection string | Required |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | OpenAI model deployment name | `gpt-35-turbo` |
| `AZURE_SEARCH_INDEX_NAME` | Search index name | `collegeproject_index` |

## Requirements

See `requirements.txt` for all dependencies:
- streamlit
- openai
- azure-search-documents==11.4.0
- azure-storage-blob
- PyPDF2
- python-docx

## Supported File Formats

- **PDF** (.pdf)
- **Word Documents** (.docx)
- **Plain Text** (.txt)

## Limitations

- The AI assistant only uses information from your uploaded documents
- Empty or image-based PDFs cannot be processed
- Maximum token limit for responses: 800 characters
- Search retrieves top 5 most relevant chunks per query

## Troubleshooting

**Error: "Could not extract text from file"**
- Ensure the file is not corrupted or image-based
- Try converting the file to a supported format

**Error: "An error occurred" when searching**
- Verify all environment variables are correctly set
- Check Azure service connectivity
- Ensure the search index exists in Azure Cognitive Search

**No results found**
- Check that documents were successfully uploaded (look for success message)
- Try rephrasing your question
- Ensure relevant content exists in your uploaded documents

## Security

- All credentials are stored as environment variables (never hardcoded)
- Documents are securely uploaded to Azure services
- No data is stored locally in the application
- Responses are generated only from your uploaded documents

## License

Specify your license here.

## Support

For issues or questions, please [create an issue](link-to-issues) or contact the development team.