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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

upload_file = st.file_uploader("Upload your PDF" , type="pdf")

if upload_file is not None and st.session_state.qa_chain is None:
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
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            retriever = retriever
        )
        st.session_state.pages = len(pages)
        st.session_state.chunks = len(chunks)
        
    st.success(f"PDF processed!! {len(pages)} pages, {len(chunks)} chunks ready.")
    
#chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write(f"**you:** {message['content']}")
    else:
        st.write(f"**AI:** {message['content']}")
            
#question input
if st.session_state.qa_chain is not None:
    question = st.text_input("Ask a question about your pdf:", key = "question input")
        
    if question and question != st.session_state.get("last_question", ""):
        with st.spinner("finding answer...."):
            answer = st.session_state.qa_chain.invoke(question)
            result = answer["result"]
    
        #save to history of the chat
        st.session_state.last_question = question
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.session_state.chat_history.append({"role": "ai", "content": result})
        
        st.rerun()
        

