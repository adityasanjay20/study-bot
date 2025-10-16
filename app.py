import streamlit as st
import os
import io
import re
from datetime import datetime
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import PyPDF2
import docx
import threading
import queue

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

# --- 2. Page Configuration ---
st.set_page_config(page_title="Study Bot AI", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# --- 3. Backend Functions ---

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
    """Extracts, chunks, and directly uploads document content to the search index with session isolation."""
    # Skip if already processed in this session
    if uploaded_file.name in st.session_state.processed_files:
        return None
    
    st.info(f"Processing '{uploaded_file.name}'...")
    
    full_text = extract_text_from_file(uploaded_file)
    if not full_text:
        st.error(f"Could not extract text from '{uploaded_file.name}'. The file might be empty or image-based.")
        return False

    text_chunks = chunk_text(full_text)
    
    documents_to_upload = []
    base_filename = os.path.splitext(uploaded_file.name)[0]
    sanitized_prefix = re.sub(r'[^a-zA-Z0-9_-]', '_', base_filename)

    for i, chunk in enumerate(text_chunks):
        # Include session_id in document ID to isolate users
        doc_id = f"{st.session_state.session_id}-{sanitized_prefix}-{i}"
        documents_to_upload.append({
            "id": doc_id,
            "content": chunk,
        })
    
    try:
        # Optimize: Upload to Azure Search in smaller batches
        batch_size = 10
        for batch_start in range(0, len(documents_to_upload), batch_size):
            batch = documents_to_upload[batch_start:batch_start + batch_size]
            search_client.upload_documents(documents=batch)
        
        # Optional: Upload to Blob Storage (can be skipped for speed)
        if BLOB_CONNECTION_STRING:
            try:
                blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
                blob_container_name = "documents"
                blob_name = f"{st.session_state.session_id}/{uploaded_file.name}"
                blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=blob_name)
                blob_client.upload_blob(uploaded_file.getvalue(), overwrite=True)
            except Exception as blob_error:
                pass  # Silently skip blob storage errors
        
        # Mark file as processed and clear outdated summaries
        st.session_state.processed_files.add(uploaded_file.name)
        st.session_state.last_upload_time = datetime.now()
        st.session_state.session_documents[uploaded_file.name] = len(text_chunks)
        st.session_state.summary = ""
        st.session_state.quiz = ""
        st.session_state.flashcards = ""
        
        st.success(f"✅ '{uploaded_file.name}' uploaded successfully! ({len(text_chunks)} chunks indexed)")
        return True
    except Exception as e:
        st.error(f"Failed to upload '{uploaded_file.name}': {e}")
        return False

def get_answer_from_ai(user_question, search_results):
    """Constructs a prompt and gets an answer from the OpenAI model."""
    context = "\n\n".join([result.get('content', '') for result in search_results])
    prompt = f"""
    You are a helpful AI study assistant. A user has asked the following question: "{user_question}"
    Using ONLY the information provided below from their study documents, answer the question clearly and concisely.
    Do not use any of your own knowledge. If the information is not in the documents, say "I couldn't find that information in your documents."
    Study Material Context:\n---\n{context}\n---
    """
    
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps people with their study documents. Be accurate, helpful, and concise."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content

def search_session_documents(search_text):
    """Search only documents from the current session."""
    try:
        # Search with a filter for current session ID
        results = search_client.search(search_text=search_text, top=5, filter=f"id eq '{st.session_state.session_id}*'")
        return list(results)
    except:
        # Fallback: search all and filter in Python (less efficient but works)
        results = search_client.search(search_text=search_text, top=20)
        session_results = [r for r in results if r['id'].startswith(st.session_state.session_id)]
        return session_results[:5]

def generate_summary_from_ai(full_text):
    """Generates a concise summary of the uploaded documents."""
    # Limit text to approximately 8000 characters (roughly 2000 tokens) to stay within context limits
    max_chars = 8000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n[Text truncated due to length...]"
    
    prompt = f"""
    Please act as an expert summarizer. Read the following text from one or more documents and generate a comprehensive summary of the key topics, concepts, and conclusions. 
    Present the summary as a list of bullet points organized by topic.

    Document Text:
    ---
    {full_text}
    ---
    """
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an expert summarization assistant. Create clear, organized, and concise summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    return response.choices[0].message.content

def generate_quiz_from_ai(full_text, num_questions=5):
    """Generates a quiz from the uploaded documents."""
    # Limit text to approximately 8000 characters to stay within context limits
    max_chars = 8000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n[Text truncated due to length...]"
    
    prompt = f"""
    Based on the following study material, generate {num_questions} multiple-choice questions that test understanding of the key concepts.
    Format each question as:
    Q#: [Question]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Answer: [Correct Letter]

    Study Material:
    ---
    {full_text}
    ---
    """
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an expert educator. Create clear, fair multiple-choice questions that test conceptual understanding."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2000
    )
    return response.choices[0].message.content

def generate_flashcards_from_ai(full_text, num_cards=10):
    """Generates flashcard content from the uploaded documents."""
    # Limit text to approximately 8000 characters to stay within context limits
    max_chars = 8000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n[Text truncated due to length...]"
    
    prompt = f"""
    Based on the following study material, generate {num_cards} flashcard pairs in the format:
    FRONT: [Key term or question]
    BACK: [Definition or answer]
    ---

    Study Material:
    ---
    {full_text}
    ---
    """
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": "You are an expert educator. Create clear, concise flashcard pairs for effective learning."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=1500
    )
    return response.choices[0].message.content

# --- 4. Streamlit UI Setup ---
st.title("🎓 Study Bot AI")
st.markdown("Your intelligent study companion powered by AI • Upload materials → Get answers, summaries, quizzes & more")

# Initialize session state
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'quiz' not in st.session_state:
    st.session_state.quiz = ""
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = ""
if 'uploaded_count' not in st.session_state:
    st.session_state.uploaded_count = 0
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'last_upload_time' not in st.session_state:
    st.session_state.last_upload_time = None
if 'session_id' not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]  # Create unique session ID
if 'session_documents' not in st.session_state:
    st.session_state.session_documents = {}  # Store docs in memory per session

# --- 5. Sidebar: Document Management ---
with st.sidebar:
    st.header("📚 Document Management")
    uploaded_files = st.file_uploader("Upload your study materials", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
    
    if uploaded_files:
        st.subheader("Processing Files...")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        success_count = 0
        total_files = len(uploaded_files)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {idx + 1}/{total_files}: {uploaded_file.name}")
            result = upload_and_index_directly(uploaded_file)
            if result is True:
                success_count += 1
            progress_bar.progress((idx + 1) / total_files)
        
        if success_count > 0:
            st.session_state.uploaded_count += success_count
            progress_bar.empty()
            status_text.empty()
            st.info(f"⏳ Please wait 2-3 seconds for the index to update, then regenerate your summaries and quizzes.")
    
    st.divider()
    
    # Study Tools Section
    st.header("🛠️ Study Tools")
    
    with st.expander("📄 Generate Summary", expanded=False):
        if st.button("Create Summary", key="summary_btn", use_container_width=True):
            with st.spinner("Creating summary from all documents..."):
                try:
                    all_results = search_session_documents("*")
                    full_text = "\n\n".join([result['content'] for result in all_results])
                    
                    if full_text:
                        st.session_state.summary = generate_summary_from_ai(full_text)
                        st.success("✅ Summary generated!")
                    else:
                        st.warning("No documents uploaded yet.")
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
    
    with st.expander("❓ Generate Quiz", expanded=False):
        quiz_count = st.slider("Number of questions", 3, 10, 5, key="quiz_count")
        if st.button("Create Quiz", key="quiz_btn", use_container_width=True):
            with st.spinner("Creating quiz..."):
                try:
                    all_results = search_session_documents("*")
                    full_text = "\n\n".join([result['content'] for result in all_results])
                    
                    if full_text:
                        st.session_state.quiz = generate_quiz_from_ai(full_text, quiz_count)
                        st.success("✅ Quiz generated!")
                    else:
                        st.warning("No documents uploaded yet.")
                except Exception as e:
                    st.error(f"Error generating quiz: {e}")
    
    with st.expander("🎯 Generate Flashcards", expanded=False):
        card_count = st.slider("Number of cards", 5, 20, 10, key="card_count")
        if st.button("Create Flashcards", key="flash_btn", use_container_width=True):
            with st.spinner("Creating flashcards..."):
                try:
                    all_results = search_session_documents("*")
                    full_text = "\n\n".join([result['content'] for result in all_results])
                    
                    if full_text:
                        st.session_state.flashcards = generate_flashcards_from_ai(full_text, card_count)
                        st.success("✅ Flashcards generated!")
                    else:
                        st.warning("No documents uploaded yet.")
                except Exception as e:
                    st.error(f"Error generating flashcards: {e}")
    
    st.divider()
    st.caption(f"📊 Session ID: `{st.session_state.session_id}`")
    st.caption(f"📄 Documents in session: {len(st.session_state.session_documents)}")
    if st.session_state.last_upload_time:
        time_since = (datetime.now() - st.session_state.last_upload_time).total_seconds()
        st.caption(f"⏱️ Last upload: {time_since:.0f} seconds ago")

# --- 6. Main Content Area ---
tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A", "📋 Summary", "❓ Quiz", "🎯 Flashcards"])

# Tab 1: Q&A
with tab1:
    st.header("Ask Your Questions")
    st.write("Search your documents for specific information")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_input("What do you want to know?", placeholder="Type your question here...")
    with col2:
        search_btn = st.button("Search", use_container_width=True)
    
    if user_question and search_btn:
        with st.spinner("🔍 Searching your documents..."):
            try:
                search_results = search_session_documents(user_question)
                
                if not search_results:
                    st.warning("⚠️ No relevant information found in your documents.")
                else:
                    answer = get_answer_from_ai(user_question, search_results)
                    st.success("✅ Answer found:")
                    st.write(answer)
                    
                    with st.expander("📌 Source Excerpts"):
                        for i, result in enumerate(search_results, 1):
                            st.caption(f"Source {i}:")
                            st.text(result['content'][:500] + "...")
            except Exception as e:
                st.error(f"Error: {e}")

# Tab 2: Summary
with tab2:
    st.header("Document Summary")
    st.write("AI-generated overview of all your uploaded materials")
    
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        if st.button("🔄 Regenerate Summary"):
            with st.spinner("Regenerating summary..."):
                try:
                    all_results = search_client.search(search_text="*", top=100)
                    full_text = "\n\n".join([result['content'] for result in all_results])
                    if full_text:
                        st.session_state.summary = generate_summary_from_ai(full_text)
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("📌 Click 'Create Summary' in the left sidebar to generate a summary of your documents.")

# Tab 3: Quiz
with tab3:
    st.header("Study Quiz")
    st.write("Test your knowledge with AI-generated questions")
    
    if st.session_state.quiz:
        st.markdown(st.session_state.quiz)
        if st.button("🔄 Generate New Quiz"):
            st.rerun()
    else:
        st.info("📌 Click 'Create Quiz' in the left sidebar to generate practice questions.")

# Tab 4: Flashcards
with tab4:
    st.header("Flashcards")
    st.write("Key terms and definitions from your materials")
    
    if st.session_state.flashcards:
        st.markdown(st.session_state.flashcards)
        if st.button("🔄 Generate New Flashcards"):
            st.rerun()
    else:
        st.info("📌 Click 'Create Flashcards' in the left sidebar to generate study cards.")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Study Bot AI v2.0")