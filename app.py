import streamlit as st
import os
import io
import re
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import PyPDF2
import docx

# --- 1. Secure Configuration and Credentials ---
# Reads all secrets from environment variables for secure deployment.
SEARCH_ENDPOINT = os.environ.get("SEARCH_ENDPOINT")
SEARCH_KEY = os.environ.get("SEARCH_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
BLOB_CONNECTION_STRING = os.environ.get("BLOB_CONNECTION_STRING")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-35-turbo")
AZURE_SEARCH_INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "collegeproject_index")

# Initialize clients for Azure services
search_creds = AzureKeyCredential(SEARCH_KEY)
openai_client = AzureOpenAI(api_version="2024-02-01", azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_KEY)
search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=search_creds)

# --- 2. Backend Functions ---

def extract_text_from_file(uploaded_file):
    """Extracts text content from an uploaded file (PDF, DOCX, or TXT)."""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(uploaded_file.getvalue()))
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    return ""

def chunk_text(text, chunk_size=5000, overlap=1000):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def upload_and_index_directly(uploaded_file):
    """Extracts, chunks, and directly uploads document content to the search index."""
    st.info(f"Processing '{uploaded_file.name}'...")
    
    full_text = extract_text_from_file(uploaded_file)
    if not full_text:
        st.error(f"Could not extract text from '{uploaded_file.name}'. The file might be empty or image-based.")
        return

    text_chunks = chunk_text(full_text)
    
    documents_to_upload = []
    base_filename = os.path.splitext(uploaded_file.name)[0]
    sanitized_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', base_filename)

    for i, chunk in enumerate(text_chunks):
        documents_to_upload.append({
            "id": f"{sanitized_prefix}-{i}",
            "content": chunk,
        })
    
    # Initialize BlobServiceClient inside the function to use the env variable
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    blob_container_name = "documents"
    blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=uploaded_file.name)
    blob_client.upload_blob(uploaded_file.getvalue(), overwrite=True)

    search_client.upload_documents(documents=documents_to_upload)
    st.success(f"'{uploaded_file.name}' is now ready to be searched!")

def get_answer_from_ai(user_question, search_results):
    """Constructs a prompt and gets an answer from the OpenAI model."""
    context = "\n\n".join([result.get('content', '') for result in search_results])
    prompt = f"""
    You are a helpful AI study assistant. A user has asked the following question: "{user_question}"
    Using ONLY the information provided below from their study documents, answer the question.
    Do not use any of your own knowledge. If the information is not in the documents, say "I couldn't find that information in your documents."
    Study Material Context:\n---\n{context}\n---
    """
    
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps people with their study documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content

# --- 3. Streamlit User Interface ---
st.title("🎓 Study Bot AI")
st.write("Upload your course materials and ask questions to get answers based on your documents.")

with st.sidebar:
    st.header("Upload Your Documents")
    uploaded_files = st.file_uploader("Choose PDF, DOCX, or TXT files", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            upload_and_index_directly(uploaded_file)

st.header("Ask a Question")
user_question = st.text_input("What do you want to know from your documents?")

if user_question:
    with st.spinner("Searching your documents and generating an answer..."):
        try:
            results = search_client.search(search_text=user_question, top=5)
            search_results = list(results) 

            if not search_results:
                st.warning("I couldn't find any relevant information in your documents to answer that question.")
            else:
                answer = get_answer_from_ai(user_question, search_results)
                st.success("Here's what I found:")
                st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")