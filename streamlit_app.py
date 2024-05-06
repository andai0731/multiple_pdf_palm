import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from pinecone import Pinecone
import os

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def generate_embeddings(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = embeddings.embed_query([chunk.page_content for chunk in text_chunks])
    return vectors

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = generate_embeddings(text_chunks)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "llm"
    
    # Upsert items into the vector store with associated embeddings
    pc.upsert(index_name, text_chunks, vectors)
    return pc

def get_conversational_chain(vector_store):
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=API_KEY, temperature=0.1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store, memory=memory)
    return conversation_chain

def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("Bot: ", message.content)

def main():
    st.set_page_config("Chat with Multiple PDFs")
    st.header("Chat with Multiple PDF 💬")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")

if __name__ == "__main__":
    main()
