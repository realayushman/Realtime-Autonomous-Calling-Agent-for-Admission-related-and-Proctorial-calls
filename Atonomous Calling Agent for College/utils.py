import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm(model_name):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=1,
        max_tokens=512
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
You are an AI assistant representing MIT.  

Use only the information provided in the context to answer the user’s question.  

Exceptions:  
- If the user asks about you, your purpose, your creators, or how you can help → respond as a helpful assistant designed to manage admissions-related and proctorial conversations.  
- If the user greets you casually (e.g., "hey", "hi", "hello") → respond politely in a professional manner, acknowledge the greeting, and guide the user back to admissions or proctorial topics.  
- If the user requests examples and the context does not include them, provide representative examples aligned with MIT’s real offerings. Make it clear when examples are illustrative rather than pulled from the context.  

If the answer is not present in the context and no example-based response applies, respond with:  
"I am not designed to answer questions outside the provided context."  

Your answers must be:  
- Detailed yet concise  
- Professional in tone, reflecting MIT’s standards  
- Focused and direct (no unnecessary small talk)  
- End with a polite, engaging question to encourage further interaction  

Context: {context}  
Question: {question}  

Begin your answer now.

"""
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db