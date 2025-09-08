'''import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
# --- Step 1: Load LLM ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"  # You can use "llama3-8b-8192" or other supported models

def load_llm(model_name):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=1,
        max_tokens=512
    )
    return llm

# --- Step 2: Custom prompt ---
CUSTOM_PROMPT_TEMPLATE = """
You are an AI assistant representing MIT.  
Use only the information provided in the context to answer the user’s question.  
If the answer is not present in the context, respond:  
"I am not designed to answer questions outside the provided context."  

Provide detailed yet concise answers, written in a professional tone that reflects MIT’s standards.  

Context: {context}  
Question: {question}  

Begin your answer directly, without greetings or small talk, and end with a polite question to engage the user further.

"""

prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# --- Step 3: Load FAISS vectorstore ---
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --- Step 4: Create RetrievalQA chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(GROQ_MODEL),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# --- Step 5: Run query ---
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})

print("RESULT:", response["result"])
#print("SOURCE DOCUMENTS:", response["source_documents"])'''


from utils import load_llm, CUSTOM_PROMPT_TEMPLATE, get_vectorstore, set_custom_prompt

GROQ_MODEL = "openai/gpt-oss-20b"  # or "llama3-8b-8192"

# Set up prompt
prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

# Load vectorstore
db = get_vectorstore()

# Create RetrievalQA chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(GROQ_MODEL),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# Run query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})

print("RESULT:", response["result"])
#print("SOURCE DOCUMENTS:", response["source_documents"])