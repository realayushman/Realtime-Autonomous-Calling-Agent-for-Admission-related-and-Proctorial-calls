import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from utils import get_vectorstore, set_custom_prompt, load_llm, CUSTOM_PROMPT_TEMPLATE
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL = "llama-3.3-70b-versatile"  
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=3,  # small window for speed
        memory_key="chat_history",
        return_messages=True
    )

def main():
    st.title("Ask Chatbot! ⚡")  # Added emoji to indicate speed

    # Load vector store once and cache it
    if 'vectorstore' not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.vectorstore = get_vectorstore()
    
    if 'qa_chain' not in st.session_state:
        with st.spinner("Initializing AI model..."):
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=load_llm(GROQ_MODEL),
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 2}),
                memory=st.session_state.memory,
                return_source_documents=False,
                combine_docs_chain_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
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
                response = st.session_state.qa_chain({"question": prompt})
                result = response["answer"]

                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()