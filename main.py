import os
import streamlit as st
import hashlib
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM

import PyPDF2
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

import requests
from fpdf import FPDF
from bs4 import BeautifulSoup
import asyncio
import aiohttp

CHROMA_PATH = "chroma"
DATA_PATH = "data"

# List of websites to crawl
urls = []

# Helper function to calculate document hash
def get_document_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Cache data with error handling for web crawling
@st.cache_data
def crawl_webpage(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text_content = ' '.join([para.get_text() for para in paragraphs])
        return text_content
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching content from {url}: {e}")
        return None

# Ensure directories exist
os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

def save_to_pdf(text, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    encoded_text = text.encode('utf-8', 'replace').decode('latin-1')
    pdf.add_page()
    pdf.multi_cell(0, 10, encoded_text)
    pdf.output(filename)

# Add to Chroma with checks for existing documents
def add_to_chroma(chunks: list[Document]):
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-m3", show_progress=True, model_kwargs={"device": "cpu"}))
    
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])

    # Avoid duplicate additions by using document hash
    new_chunks = []
    for chunk in chunks:
        chunk_hash = get_document_hash(chunk.page_content)
        if chunk.metadata["id"] not in existing_ids:
            chunk.metadata["hash"] = chunk_hash
            new_chunks.append(chunk)

    if len(new_chunks):
        db.add_documents(new_chunks, ids=[chunk.metadata["id"] for chunk in new_chunks])
        db.persist()

# Extract text from PDF files with error handling
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

# Asynchronous web crawling
async def fetch_url(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                soup = BeautifulSoup(await response.text(), 'html.parser')
                paragraphs = soup.find_all('p')
                text_content = ' '.join([para.get_text() for para in paragraphs])
                return text_content
    except Exception as e:
        st.error(f"Error while fetching {url}: {e}")
        return None

async def crawl_urls(urls):
    results = await asyncio.gather(*[fetch_url(url) for url in urls])
    return results

# Adding URL and PDF input interface
def add_url_and_pdf_input():
    st.header("Upload PDFs or Add URLs for Content Extraction")

    # Create two columns for URLs and PDFs
    col1, col2 = st.columns(2)

    with col1:
        url1 = st.text_input("Enter URL 1:")
        url2 = st.text_input("Enter URL 2:")

    with col2:
        pdf1 = st.file_uploader("Upload PDF 1", type="pdf")
        pdf2 = st.file_uploader("Upload PDF 2", type="pdf")

    if st.button("Submit"):
        # Process URLs
        urls = [url1, url2]
        pdf_files = [pdf1, pdf2]

        with st.spinner("Processing URLs and PDFs..."):
            # Crawl URLs asynchronously
            url_results = asyncio.run(crawl_urls([url for url in urls if url]))
            for i, website_text in enumerate(url_results):
                if website_text:
                    pdf_path = f"data/url_content_{i+1}.pdf"
                    save_to_pdf(website_text, pdf_path)
                    doc = Document(page_content=website_text, metadata={"source": urls[i]})
                    add_to_chroma([doc])
                    st.success(f"Content from {urls[i]} uploaded successfully!")

            # Process PDFs
            for i, pdf in enumerate(pdf_files):
                if pdf:
                    pdf_path = f"data/uploaded_pdf_{i+1}.pdf"
                    with open(pdf_path, "wb") as f:
                        f.write(pdf.getbuffer())

                    pdf_text = extract_text_from_pdf(pdf_path)
                    doc = Document(page_content=pdf_text, metadata={"source": pdf_path})
                    add_to_chroma([doc])
                    st.success(f"Content from PDF {i+1} uploaded successfully!")

def main():
    st.title('Welcome to SourceMind: Your AI Document Search Assistant')

    st.markdown("""
    This tool allows you to upload PDF files or provide URLs for content extraction. 
    Once uploaded, you can ask questions about the content, and the AI will provide relevant answers based on the documents you have uploaded.
    """, unsafe_allow_html=True)

    # URL and PDF input section
    add_url_and_pdf_input()

    # User input for question
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question:
            db = Chroma(persist_directory=CHROMA_PATH,
                        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-m3", show_progress=True, model_kwargs={"device": "cpu"}))

            results = db.similarity_search_with_relevance_scores(question, k=2)

            if results:
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

                prompt = ChatPromptTemplate.from_template("""
                Answer the question based only on the following context:

                {context}

                ---

                Answer the question based on the above context: {question}
                """).format(context=context_text, question=question)

                model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
                response_text = model.predict(prompt)

                st.write(f"### Answer: {response_text}")
                st.write(f"### Sources: {[doc.metadata.get('source') for doc, _ in results]}")

            else:
                st.warning("No relevant content found in the uploaded documents. Please upload more content or refine your question.")
        else:
            st.error("Please enter a question to ask!")

# Run the main function to launch the app
if __name__ == '__main__':
    main()
