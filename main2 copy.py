import os
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# === LangChain dan Chroma ===
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# === Konfigurasi Token dan Key ===
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-safeguard-20b")


# === Siapkan Folder DB dan Data ===
CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"



# === Load Vectorstore (gunakan yang sudah dibuild dari build_dataset.py) ===
def get_vectorstore():
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        raise Exception("❌ ChromaDB belum ada. Jalankan 'build_dataset.py' dulu untuk membuat dataset.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    print("✅ ChromaDB berhasil dimuat.")
    return vectordb


# === Inisialisasi LangChain LLM + Chain ===
def create_chain():
    # Inisialisasi LLM Groq
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-safeguard-20b",
        temperature=0.3,
        # reasoning_effort="medium"  # memberi ruang untuk berpikir lebih logis
    )

    # Load ChromaDB
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Template Prompt
    prompt = ChatPromptTemplate.from_template("""
    Kamu adalah asisten virtual yang ramah, informatif, dan sopan. 
    Tugasmu adalah menjawab pertanyaan pengguna berdasarkan informasi yang ada di dokumen berikut.

    Gunakan **bahasa Indonesia yang jelas dan mudah dipahami**.

    ---

    ### 🎯 ATURAN MENJAWAB

    1. **Fokus utama kamu adalah memberikan jawaban berdasarkan data dari dokumen.**
    - Jika informasi ditemukan sebagian (misalnya hanya harga per hari), gunakan data tersebut untuk menjawab sebaik mungkin.
    - Jangan menjawab dengan “tidak tahu” jika masih ada informasi yang relevan, meskipun sebagian.

    2. **Pertanyaan umum tentang harga, paket, atau layanan:**
    - Jika dokumen berisi beberapa pilihan paket (misalnya trip, sewa, atau layanan lainnya),
        tampilkan semua pilihan yang relevan dengan nama dan harganya.
    - Gunakan format daftar agar mudah dibaca.
    - Contoh:
        "Ada beberapa paket trip Bromo yang tersedia:
        - Sewa Jeep Only: Rp 1.200.000 / jeep
        - Open Trip: Rp 350.000 / orang
        - Private Trip: Rp 1.700.000 / trip (maksimal 6 orang)."

    3. **Pertanyaan dengan perhitungan sederhana (misalnya harga untuk beberapa hari, orang, atau unit):**
    - Lakukan perhitungan logis berdasarkan harga yang ada dalam dokumen.
    - **JANGAN tampilkan proses perhitungannya secara rinci (seperti “Rp 1.250.000 x 2”).**
    - Cukup tampilkan hasil akhirnya secara singkat dan alami.
    - Contoh:
        "Harga sewa Hi Ace Premio untuk 2 hari adalah Rp 2.500.000."

    4. **Pertanyaan lanjutan (seperti 'kalau nambah 1 hari lagi?'):**
    - Gunakan konteks dari percakapan sebelumnya jika memungkinkan.
    - Tambahkan perhitungan baru berdasarkan harga sebelumnya, **tapi hanya tampilkan hasil akhirnya.**

    5. **Pertanyaan tentang gambar, foto, atau tautan:**
    - Jika di dokumen terdapat URL atau tautan gambar, tampilkan link tersebut dengan kalimat yang sopan.
    - Contoh:
        "Berikut tautan gambarnya: https://contoh.com/gambar.jpg"

    6. **Jika informasi benar-benar tidak tersedia dalam dokumen:**
    - Jawab dengan kalimat:
        "Maaf, saya tidak menemukan informasi terkait di dokumen ini. Berikut isi dokumen yang saya ketahui: {context}"

    7. **Jika pertanyaan menyebut 'perhari' atau 'per hari':**
    - Carilah informasi harga yang mengandung kata 'per hari' atau '/ hari' dalam dokumen.

    ---

    ### 📘 KONTEKS DOKUMEN
    {context}

    ### ❓PERTANYAAN
    {question}
    """)



    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain


# === Buat objek chain global ===
chain = create_chain()


# === Handler Command /start ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Halo 👋! Saya Chatbot Virtual Assistant Travel Malang ID. Silahkan tanyakan apa saja seputar travel di Malang."
    )


# === Handler Pesan User ===
# Memory sederhana untuk menyimpan konteks tiap user
user_memory = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    user_id = update.effective_chat.id

    print(f"📩 Pesan diterima dari {user_id}: {user_text}")
    await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)

    # Ambil riwayat percakapan sebelumnya (kalau ada)
    previous_context = user_memory.get(user_id, "")

    try:
        # Gabungkan pesan baru dengan konteks lama agar LLM tahu riwayatnya
        full_input = f"{previous_context}\n\nPengguna: {user_text}"

        response = chain.invoke(full_input)
        answer = response.content.strip()

        print(f"💬 Bot menjawab: {answer}")
        await update.message.reply_text(answer)

        # Simpan konteks terbaru (maksimal 5 percakapan terakhir)
        new_context = f"{previous_context}\nPengguna: {user_text}\nBot: {answer}"
        user_memory[user_id] = "\n".join(new_context.splitlines()[-10:])  # batasi agar tidak terlalu panjang

    except Exception as e:
        print(f"❌ Error: {e}")
        await update.message.reply_text("⚠️ Maaf, terjadi kesalahan saat memproses pesan Anda.")



# === Jalankan Bot ===
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot sedang berjalan dengan LangChain + ChromaDB...")
    app.run_polling()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Bot dihentikan oleh pengguna.")
