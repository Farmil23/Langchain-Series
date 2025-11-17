import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Muat environment variables
load_dotenv()

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="RAG Document Q&A on Google Adsense", layout="wide")
st.title("Query Your Google adsense FAQ 📚")

# Ambil API keys dari environment
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Tampilkan peringatan jika API keys tidak ditemukan
if not groq_api_key:
    st.warning("GROQ_API_KEY not found. Please add it to your .env file.")
if not openai_api_key:
    st.warning("OPENAI_API_KEY not found. Please add it to your .env file.")
    
# Set API keys di environment (beberapa library mungkin memerlukannya)
os.environ["GROQ_API_KEY"] = groq_api_key or ""
os.environ["OPENAI_API_KEY"] = openai_api_key or ""

# Inisialisasi LLM
# Pastikan API key ada sebelum mencoba inisialisasi
if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
else:
    llm = None
    st.error("Groq LLM cannot be initialized. Please check your GROQ_API_KEY.")
    
    
# Template Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Sedang membuat Vector embedding ..... Tunggu sebentar!"):
            try:
                pdf_directory = "AI_ASSISTENT_EARN/books"
                
                if not os.path.exists(pdf_directory):
                    st.error(f"Directory {pdf_directory} tidak dapat ditemukan!")
                    return
                
                if not openai_api_key:
                    st.error("OpenAI API key is missing. Cannot create embeddings.")
                    return
                
                st.session_state.embeddings= OpenAIEmbeddings()
                st.session_state.loader = PyPDFDirectoryLoader(pdf_directory)
                st.session_state.docs = st.session_state.loader.load()
                
                if not st.session_state.docs:
                    st.error("File dokumen tidak dapat ditemukan!")
                    return
                
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
                st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                st.success("Vector Berhasil Dibuat!")
                
            except Exception as e:
                st.error(f"An error occurred during embedding creation: {e}")
                if "vectors" in st.session_state:
                    del st.session_state.vectors # Hapus state jika gagal
    else:
        st.info("Vector store already loaded.")
        
if st.button("Create Vector Embeddings"):
    create_vector_embedding()

user_prompt = st.text_input("Masukkan Query tentang Google adsense")

if user_prompt:
    try:
        if "vectors" not in st.session_state:
            st.warning(*"Please create the vector by clicking embeddings button!")
        elif not llm:
            st.error("LLM is not initialized. Cannot process query.")
        else:
            with st.spinner("Searching the answer"):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                start = time.process_time()
                response = retrieval_chain.invoke({'input' : user_prompt})
                response_time = time.process_time() - start
                
                st.write(response['answer'])
                st.write(f"Response time : {response_time}")
                
                with st.expander("Show the context"):
                    for i, doc in enumerate(response['context']):
                        st.write(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')})**")
                        st.write(doc.page_content)
                        st.write("-" * 20)
            
    except Exception as e:
        st.error(f"An error occurred during query: {e}")