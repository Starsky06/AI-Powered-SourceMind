import os
import streamlit as st
import hashlib
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
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

# FAISS
import faiss
import numpy as np

# Define paths
FAISS_PATH = "faiss_index"  # FAISS index will be stored here
DATA_PATH = "data"

# Helper function to calculate document hash
def get_document_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

# Cache data with error handling for web crawling
# Removed @st.cache_data decorator from async functions
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
os.makedirs(FAISS_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

def save_to_pdf(text, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.add_page()

    # Handle large text by adding text until it overflows
    while text:
        # Replace unsupported characters with a placeholder or remove them
        safe_text = text[:1000].encode('latin-1', 'replace').decode('latin-1')  # Replace non-latin characters
        pdf.multi_cell(0, 10, safe_text)
        text = text[1000:]  # Remove the text that was already added
    pdf.output(filename)


# Initialize FAISS index and Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
# Initialize FAISS index and Embeddings
embedding_size = 1024  # Update the embedding size to match the output dimensionality

# FAISS index creation function with persistence
def create_faiss_index():
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
    else:
        index = faiss.IndexFlatL2(embedding_size)  # Use the updated embedding size
    return index



# Add document embeddings to FAISS index
def add_to_faiss(index, chunks: list[Document]):
    embeddings = []
    ids = []
    for chunk in chunks:
        chunk_embedding = embedding_function.embed_documents([chunk.page_content])[0]
        embeddings.append(chunk_embedding)
        ids.append(chunk.metadata["id"])
    
    embeddings = np.array(embeddings).astype('float32')  # Convert to the correct type for FAISS

    # Debug: Print the shape of embeddings
    print("Embedding shape:", embeddings.shape)

    # Ensure correct shape for FAISS
    assert embeddings.shape[1] == embedding_size, f"Embedding dimensionality mismatch: {embeddings.shape[1]} != {embedding_size}"

    index.add(embeddings)  # Add to FAISS index



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
    return [result for result in results if result]  # Filter out None values

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
            faiss_index = create_faiss_index()
            for i, website_text in enumerate(url_results):
                if website_text:
                    pdf_path = f"data/url_content_{i+1}.pdf"
                    save_to_pdf(website_text, pdf_path)
                    doc = Document(page_content=website_text, metadata={"source": urls[i], "id": get_document_hash(website_text)})
                    add_to_faiss(faiss_index, [doc])  # Add to FAISS
                    st.success(f"Content from {urls[i]} uploaded successfully!")

            # Process PDFs
            for i, pdf in enumerate(pdf_files):
                if pdf:
                    pdf_path = f"data/uploaded_pdf_{i+1}.pdf"
                    with open(pdf_path, "wb") as f:
                        f.write(pdf.getbuffer())

                    pdf_text = extract_text_from_pdf(pdf_path)
                    doc = Document(page_content=pdf_text, metadata={"source": pdf_path, "id": get_document_hash(pdf_text)})
                    add_to_faiss(faiss_index, [doc])  # Add to FAISS
                    st.success(f"Content from PDF {i+1} uploaded successfully!")

            # Save the FAISS index after adding documents
            faiss.write_index(faiss_index, FAISS_PATH)

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
            faiss_index = create_faiss_index()

            # Query the FAISS index (assuming the documents were added beforehand)
            query_embedding = embedding_function.embed(question)
            D, I = faiss_index.search(np.array([query_embedding]), k=5)  # k=5 for top 5 results

            # Show the top 5 most similar documents
            st.write(f"### Top 5 Results:")
            for idx in I[0]:
                doc = faiss_index.reconstruct(idx)
                st.write(doc.metadata["source"])  # Display the source of the document
        else:
            st.error("Please enter a question to ask!")

# Run the main function to launch the app
if __name__ == '__main__':
    main()
