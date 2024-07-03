import streamlit as st
import requests
import numpy as np
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import MarkdownTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import re
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.schema.document import Document
import pickle

# Load environment variables
GOOGLE_API_KEY = "AIzaSyCzYNTSe-ZSYPINi_0L4ElcYKX-ffWXsrI"

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

def get_data_from_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch data from {url}")
        return None, None

    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    metadata = {'url': url}

    return text, metadata

def store_docs(text, metadata):
    # Generate embeddings for the documents
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_chunks = text_to_docs(text, metadata)
    embeddings = [embedding_model.encode(doc.page_content) for doc in doc_chunks]

    # Convert embeddings to numpy array
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # Create a Faiss index
    dimension = embeddings_np.shape[1]  # Dimension of the embeddings
    index = faiss.IndexFlatIP(dimension)  # Using IndexFlatIP for inner product similarity search
    index.add(embeddings_np)

    # Create a dictionary to store documents
    doc_dict = {i: doc for i, doc in enumerate(doc_chunks)}

    # Use HuggingFaceEmbeddings for the embedding function
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vector store with index_to_docstore_id argument
    index_to_docstore_id = {i: i for i in range(len(doc_chunks))}
    faiss_vector_store = FAISS(embedding_function=hf_embeddings, index=index, docstore=doc_dict,
                               index_to_docstore_id=index_to_docstore_id)

    # Save the FAISS index and doc_dict to disk
    faiss.write_index(index, "faiss_index")
    with open("doc_dict.pkl", "wb") as f:
        pickle.dump(doc_dict, f)

    # Optionally, you can return the Faiss index and use it later
    return index, faiss_vector_store

def merge_hyphenated_words(text):
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def fix_newlines(text):
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

def remove_multiple_newlines(text):
    return re.sub(r"\n{2,}", "\n")

def clean_text(text):
    cleaning_functions = [merge_hyphenated_words, fix_newlines, remove_multiple_newlines]
    for cleaning_function in cleaning_functions:
        text = cleaning_function(text)
    return text

def text_to_docs(text, metadata):
    doc_chunks = []
    text_splitter = MarkdownTextSplitter(chunk_size=2048, chunk_overlap=128)
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        doc = Document(page_content=chunk, metadata=metadata)
        doc_chunks.append(doc)
    return doc_chunks

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def main():
    st.set_page_config("Website Chatbot")
    st.title("Website Chatbot")

    url = st.text_input("Enter the URL of the website:")

    if st.button("Fetch Data"):
        text, metadata = get_data_from_website(url)
        if text is not None and metadata is not None:
            store_docs(text, metadata)
            st.success("Data fetched and stored successfully!")

    user_question = st.text_input("Ask a question:")
    if st.button("Get Answer"):
        if not os.path.exists("faiss_index") or not os.path.exists("doc_dict.pkl"):
            st.error("Vector store not found. Please fetch data first.")
        else:
            with st.spinner("Generating answer..."):
                # Load the FAISS index and document dictionary
                index = faiss.read_index("faiss_index")
                with open("doc_dict.pkl", "rb") as f:
                    doc_dict = pickle.load(f)

                # Use HuggingFaceEmbeddings for the embedding function
                hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Create FAISS vector store from the loaded index and document dictionary
                index_to_docstore_id = {i: i for i in range(len(doc_dict))}
                faiss_vector_store = FAISS(embedding_function=hf_embeddings, index=index, docstore=doc_dict,
                                           index_to_docstore_id=index_to_docstore_id)

                # Perform a similarity search to retrieve the most relevant documents
                question_embedding = hf_embeddings.embed_query(user_question)
                scores, retrieved_indexes = index.search(np.array([question_embedding]), k=5)
                input_documents = [doc_dict[int(idx)] for idx in retrieved_indexes[0] if int(idx) in doc_dict]

                # input_documents = [doc_dict[int(idx)] for idx in retrieved_indexes[0]]

                chain = get_conversational_chain()
                answer = chain({"input_documents": input_documents, "question": user_question}, return_only_outputs=True)
                st.write("Answer:", answer["output_text"])

if __name__ == "__main__":
    main()
