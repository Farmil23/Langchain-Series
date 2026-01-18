
# 🦜🔗 LangChain Series: From Zero to Hero

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-orange?style=for-the-badge&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

> **Repositori ini berisi dokumentasi perjalanan belajar, tutorial, dan proyek praktis dalam menguasai LangChain untuk membangun aplikasi berbasis LLM (Large Language Models).**

Di sini Anda akan menemukan berbagai materi mulai dari pengenalan dasar, teknik *prompt engineering*, penggunaan *Vector Stores*, hingga implementasi arsitektur RAG (*Retrieval-Augmented Generation*) pada dokumen nyata.

---

## 📂 Struktur Pembelajaran

Repositori ini dibagi menjadi beberapa modul dan proyek berdasarkan topik pembahasan:

### 1. 🧠 Dasar-Dasar LangChain (`/LangChain`)
Modul ini mencakup konsep fundamental untuk memulai pengembangan aplikasi AI.
* **Pengenalan OpenAI**: Cara menghubungkan dan memanggil model LLM (`1.1OpenAi`).
* **Embeddings**: Memahami representasi vektor dari teks (`openai-embedding`).
* **Data Ingestion**: Teknik memuat berbagai format data (PDF, TXT, XML) agar dapat dibaca oleh AI (`3.2-DataIngestion`).
* **Vector Stores (FAISS)**: Implementasi penyimpanan vektor lokal untuk pencarian semantik yang cepat (`faiss`).

### 2. 🛠️ Proyek Praktis
Implementasi konsep ke dalam aplikasi nyata:

* **🤖 ConverChatbotQA**
    * Membangun chatbot percakapan yang mampu menjawab pertanyaan (QA) dengan konteks tertentu.
    * *Teknologi*: Conversational Chains, Memory.

* **📄 DOCUMENT_PROJECT (RAG System)**
    * Aplikasi untuk berinteraksi dengan dokumen (seperti Jurnal Ilmiah/Paper).
    * Sistem membaca file PDF (contoh: *Attention Is All You Need*, *LLM Papers*), mengubahnya menjadi vektor, dan memungkinkan pengguna "bertanya" pada dokumen tersebut.
    * *Teknologi*: PyPDFLoader, RAG Architecture.

* **💰 AI_ASSISTENT_EARN**
    * Proyek asisten AI yang berfokus pada analisis atau pengolahan informasi dari buku/laporan (contoh: *The AdSense Report*).

* **🔎 Vector Retriever (`/vectorretriever`)**
    * Eksperimen mendalam mengenai cara kerja *retriever* untuk mengambil informasi relevan dari database vektor.

* **🚀 Mini Projects (`Project-1`, `Project-2`, `Project-3`)**
    * Latihan bertahap membangun aplikasi LLM sederhana menggunakan Jupyter Notebook.

---

## 🛠️ Teknologi yang Digunakan

* **Python**: Bahasa pemrograman utama.
* **LangChain**: Framework orkestrasi LLM.
* **OpenAI API**: Penyedia model bahasa (GPT-3.5/GPT-4).
* **FAISS (Facebook AI Similarity Search)**: Library untuk pencarian similaritas vektor yang efisien.
* **PyPDF**: Untuk ekstraksi teks dari file PDF.
* **Jupyter Notebook**: Lingkungan interaktif untuk eksperimen kode.

---

## 🚀 Cara Memulai

Ikuti langkah-langkah ini untuk menjalankan materi di komputer lokal Anda:

1.  **Clone Repositori**
    ```bash
    git clone [https://github.com/username-anda/langchain-series.git](https://github.com/username-anda/langchain-series.git)
    cd langchain-series
    ```

2.  **Buat Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependensi**
    Pastikan Anda menginstall semua library yang dibutuhkan.
    ```bash
    pip install -r requirements.txt
    ```
    *(Catatan: Jika folder proyek memiliki `requirements.txt` terpisah, install sesuai kebutuhan proyek tersebut).*

4.  **Konfigurasi API Key**
    Buat file `.env` di root folder dan masukkan API Key OpenAI Anda:
    ```env
    OPENAI_API_KEY=sk-ur_api_key_here
    ```

5.  **Jalankan Notebook/Script**
    Buka Jupyter Notebook untuk memulai belajar:
    ```bash
    jupyter notebook
    ```

---

## 🤝 Kontribusi

Repositori ini bersifat *open-source* dan terbuka untuk kontribusi. Jika Anda ingin menambahkan materi baru, memperbaiki *bug*, atau meningkatkan dokumentasi:
1.  Fork repositori ini.
2.  Buat branch fitur baru.
3.  Submit Pull Request.

---

<div align="center">
  <small>Happy Coding & Learning AI! 🚀</small>
</div>

```
