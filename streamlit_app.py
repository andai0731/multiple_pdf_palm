import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

from dotenv import load_dotenv
#import pinecone
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

API = os.getenv("GOOGLE_API_KEY")
PINECONE = os.getenv("PINECONE_API_KEY")
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
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=API, temperature=0.1)
    embeddings_list = [llm.embed(text) for text in text_chunks]
    return embeddings_list



def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vectors = generate_embeddings(text_chunks)
    # Initialize Pinecone
    #Pinecone(api_key=PINECONE)  # Use your Pinecone API key
    pc = Pinecone(api_key="081c6b89-ff28-4673-9a4b-5912b5cfcff3")
    index = pc.Index("llm")
    vector_store = index

    embeddings_list = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
    # Upsert items into the vector store with associated embeddings
    vector_store.upsert(items=text_chunks, ids=range(len(text_chunks)), vectors=embeddings_list)
    return vector_store

def get_conversational_chain(vector_store):
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key=API, temperature=0.1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
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
