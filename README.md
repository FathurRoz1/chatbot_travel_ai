# Zenith Travel AI 🤖🌍

Zenith Travel AI adalah asisten virtual berbasis Telegram yang dirancang khusus untuk memberikan informasi, rekomendasi, dan panduan perjalanan (travel) di area Malang. Aplikasi ini menggunakan teknologi **RAG (Retrieval-Augmented Generation)** dengan **LangChain**, **ChromaDB**, dan ditenagai oleh **Groq LLM** agar mampu menjawab pertanyaan pengguna berdasarkan dataset lokal secara cerdas.

## 🌟 Fitur Utama
- **Telegram Bot Interface:** Interaksi mudah dan cepat melalui Telegram.
- **RAG (Retrieval-Augmented Generation):** Mampu menjawab berdasarkan dokumen (PDF/TXT) yang ditambahkan ke dalam folder `data/`.
- **Fast Inference:** Menggunakan API Groq yang sangat cepat untuk pemrosesan LLM.
- **Local Vector Database:** Menggunakan ChromaDB dan HuggingFace Embeddings (`all-MiniLM-L6-v2`) untuk pencarian semantik dokumen secara lokal (ramah memori dan gratis).
- **Chat Logging:** Menyimpan riwayat percakapan pengguna ke dalam database PostgreSQL.
- **Dataset API:** Terdapat modul API terpisah untuk mengelola dataset secara dinamis.

## 🛠️ Prasyarat
Sebelum menginstal aplikasi ini, pastikan Anda telah memiliki:
- **Python 3.10+**
- **PostgreSQL** (untuk database log chat)
- **Telegram Bot Token** (didapatkan dari [@BotFather](https://t.me/BotFather))
- **Groq API Key** (didapatkan dari [Groq Console](https://console.groq.com/))

---

## 🚀 Panduan Setup & Instalasi

### 1. Clone & Masuk ke Direktori
Jika Anda belum melakukannya, buka terminal dan masuk ke direktori proyek ini:
```bash
cd zenith_travel_ai
```

### 2. Buat Virtual Environment (Opsional tapi Direkomendasikan)
Buat dan aktifkan virtual environment agar dependensi Python tidak bentrok dengan proyek lain:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Dependensi
Jalankan perintah berikut untuk menginstal semua *library* yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment Variables (.env)
Salin file konfigurasi contoh dan ubah namanya menjadi `.env`:
```bash
cp ".env example" .env
```
Buka file `.env` menggunakan teks editor Anda dan isi nilai variabelnya:
```ini
# Token dan API Key
TELEGRAM_TOKEN=your_telegram_bot_token_here
GROQ_API_KEY=your_groq_api_key_here

# Model LLM yang digunakan
MODEL_NAME=llama3-70b-8192 # atau model Groq lainnya

# Database PostgreSQL
DB_HOST=127.0.0.1
DB_NAME=zenith_travel
DB_USER=postgres
DB_PASSWORD=your_database_password_here

DATASET_API_TOKEN=your_dataset_api_token_here
DATASET_BASE_DIR=D:/Project Python/zenith_travel_ai/
```

### 5. Siapkan Database
1. Buka PostgreSQL (pgAdmin / psql).
2. Buat database baru bernama `zenith_travel` (sesuai nama di `DB_NAME`).

### 6. Membangun Dataset (VectorDB)
Aplikasi ini menggunakan dokumen eksternal untuk menjawab. 
1. Masukkan file informasi travel berformat **.txt** atau **.pdf** ke dalam folder `data/`.
2. Jalankan script berikut untuk memproses dokumen ke dalam ChromaDB:
```bash
python build_dataset.py
```
*Tunggu hingga proses ekstraksi dan embedding selesai.*

---

## 🏃‍♂️ Menjalankan Aplikasi

### Menjalankan Bot Telegram
Untuk menghidupkan bot utama, jalankan:
```bash
python main2.py
```
Jika berhasil, bot sudah aktif dan dapat di-chat di Telegram.

### Menjalankan Dataset API (Opsional)
Jika Anda membutuhkan endpoint untuk menambah/menghapus dataset melalui API (misalnya dihubungkan ke dashboard web), Anda bisa menjalankan:
```bash
python dataset_api.py
```

## 📚 Struktur Folder Penting
- `data/` : Tempat menaruh file PDF/TXT untuk dijadikan otak chatbot.
- `chroma_db/` : Database vektor lokal yang digenerate oleh `build_dataset.py`.
- `main2.py` : Script utama (entry-point) untuk bot Telegram.
- `build_dataset.py` : Script pembangun vector database.
- `chatlog_db.py` : Modul koneksi ke PostgreSQL.
- `prompt_template.py` : Tempat mengatur *prompt* / instruksi dasar bagi LLM.

---
**Dibuat dengan ❤️ untuk kemajuan pariwisata Malang!**
