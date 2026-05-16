# RAG-based PDF Q&A App

An AI-powered application that lets you upload any PDF and ask questions about it in natural language.

## How it works
- Upload any PDF document
- Ask questions in plain English
- AI retrieves relevant sections and answers using Google Gemini

## Tech Stack
- Python
- LangChain
- Google Gemini API
- FAISS Vector Database
- Streamlit

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Google Gemini API key to `.env` file as `GOOGLE_API_KEY`
4. Run: `streamlit run app.py`