import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
import tempfile

load_dotenv()

st.title("PDF Question Answering App")
st.write("Upload a PDF ad ask question about it")

upload_file = st.file_uploader("Upload your PDF" , type="pdf")

if upload_file is not None:
    with tempfile.NamedTemporaryFile(delete = False, suffix=".pdf") as tmp:
        tmp.write(upload_file.read())
        tmp_path = tmp.name
        
    with st.spinner("Processing your PDF...."):
        # load
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        #chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        chunks = splitter.split_documents(pages)
        
        #embed+store
        
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k':3})
        
        #llm
        llm = GoogleGenerativeAI(model="gemini-2.5-flash")
        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            retriever = retriever
        )
    st.success(f"PDF processed!! {len(pages)} pages, {len(chunks)} chunks ready.")
    
    question = st.text_input("Ask a question about your PDF:")
    
    if question:
        with st.spinner("Finding answer....."):
            answer = qa_chain.invoke(question)
        st.write("**Answer:**")
        st.write(answer['result'])
    
        

