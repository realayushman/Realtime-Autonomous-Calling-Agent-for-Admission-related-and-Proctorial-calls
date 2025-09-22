import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from utils import get_vectorstore, set_custom_prompt, load_llm, CUSTOM_PROMPT_TEMPLATE
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL = "openai/gpt-oss-20b"  
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def main():
    st.title("Ask Chatbot! ⚡")  # Added emoji to indicate speed

    # Load vector store once and cache it
    if 'vectorstore' not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.vectorstore = get_vectorstore()
    
    if 'qa_chain' not in st.session_state:
        with st.spinner("Initializing AI model..."):
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(GROQ_MODEL),
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 2}),  # Reduced from 3 to 2
                return_source_documents=False,  # Disable if not showing sources
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Show loading indicator
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.qa_chain.invoke({'query': prompt})
                result = response["result"]

                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()