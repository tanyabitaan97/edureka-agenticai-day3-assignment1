import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain (Edureka-compatible)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# -------------------------------------------------
# ENVIRONMENT SETUP
# -------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "models/text-bison-001"

st.set_page_config(
    page_title="AI Legal Document Review Assistant",
    layout="wide"
)

# -------------------------------------------------
# SIDEBAR – CONFIGURATION
# -------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found in environment variables")
        st.stop()
    else:
        st.success("API Key Loaded Successfully")

    st.markdown("---")
    st.subheader("📄 Upload Documents")

    pdf_docs = st.file_uploader(
        "Upload text-based PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    process_button = st.button("Process Documents")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def extract_pdf_text(pdf_files):
    """Extract text from PDFs using PyPDF2"""
    full_text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        except Exception:
            st.warning(f"Could not read file: {pdf.name}")
    return full_text


def split_text_into_chunks(text):
    """Split text into chunks"""
    if not text or text.strip() == "":
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vector_store(chunks):
    """Create FAISS vector store using local embeddings"""
    if not chunks:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embedding=embeddings)


def get_qa_chain(vector_store):
    """Create RetrievalQA chain"""
    llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )


def generate_summary_any_document(vector_store):
    """
    Generate a high-quality summary for ANY type of document
    using Gemini 2.5 Flash.
    """

    retriever = vector_store.as_retriever(search_kwargs={"k": 12})
    docs = retriever.get_relevant_documents("Summarize the document")

    if not docs:
        return "No content available for summarization."

    # Combine retrieved content
    combined_text = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
You are an expert document analyst.

Summarize the following document clearly and concisely.
Adapt the summary to the document type automatically.

Your summary should include:
- Purpose of the document
- Key topics or sections
- Important details, facts, or decisions
- Any conclusions, outcomes, or next steps (if applicable)

Use clear headings and bullet points where appropriate.

Document Content:
{text}
"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    summary = chain.run(text=combined_text)

    return summary



# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "text_preview" not in st.session_state:
    st.session_state.text_preview = ""

# -------------------------------------------------
# DOCUMENT PROCESSING
# -------------------------------------------------
if process_button:
    if not pdf_docs:
        st.warning("Please upload at least one PDF file.")
    else:
        with st.spinner("Extracting and processing documents..."):
            raw_text = extract_pdf_text(pdf_docs)
            chunks = split_text_into_chunks(raw_text)

            if not chunks:
                st.error(
                    "No readable text found. "
                    "Please upload text-based (non-scanned) PDFs."
                )
            else:
                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.text_preview = raw_text[:2000]
                st.success("Documents processed successfully!")

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title("⚖️ AI Legal Document Review Assistant")

if st.session_state.text_preview:
    with st.expander("📄 PDF Text Preview"):
        st.text(st.session_state.text_preview)

import re

# -------------------------------------------------
# QUESTION ANSWERING
# -------------------------------------------------
user_question = st.chat_input("Ask a question about the uploaded legal documents")

if user_question:
    st.chat_message("human").markdown(user_question)

    qa_chain = get_qa_chain(st.session_state.vector_store)
    response = qa_chain.run(user_question) 
    st.chat_message("ai").markdown(response)

# -------------------------------------------------
# SUMMARY SECTION
# -------------------------------------------------
if st.session_state.vector_store:
    st.markdown("---")
    if st.button("📑 Generate Document Summary"):
        with st.spinner("Generating summary..."):
            summary = generate_summary_any_document(st.session_state.vector_store)
            st.subheader("Document Summary")
            st.markdown(summary)
