import os
from dotenv import load_dotenv

# Impor komponen yang lebih modern dari LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Muat API Key dari file .env
load_dotenv()

# 1. Inisialisasi Model LLM dengan nama yang benar
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 2. Buat Prompt Template yang lebih sesuai untuk Chatbot
# Di sini kita definisikan peran AI di bagian "system"
prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah asisten AI pribadi untuk Farhan. Jawab semua pertanyaan dengan ramah dan membantu."),
    ("user", "{question}")
])

# 3. Inisialisasi Output Parser
# Ini akan mengubah output dari AI menjadi teks biasa yang mudah dibaca
output_parser = StrOutputParser()

# 4. Buat Rantai (Chain) dengan cara modern (LCEL)
# Tanda | (pipe) ini seperti "rantai" yang menghubungkan komponen
chain = prompt | llm | output_parser

print("🤖 Chatbot Asisten Farhan Aktif!")
print("Ketik 'exit' atau 'keluar' untuk berhenti.")

# Loop agar chatbot bisa berjalan terus-menerus
while True:
    # Menggunakan nama variabel yang baik (bukan 'input')
    user_question = input("\nAnda: ")
    
    if user_question.lower() in ["exit", "keluar"]:
        print("🤖 Sampai jumpa, Farhan!")
        break
    
    # Jalankan rantai dengan pertanyaan pengguna
    response = chain.invoke({"question": user_question})
    
    print(f"Asisten: {response}")