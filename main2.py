import os
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv
from chatlog_db import save_chatlog

# === LangChain dan Chroma ===
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq


# === Import Prompt Template ===
from prompt_template import get_prompt

# === Load file .env ===
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

CHROMA_PATH = "./chroma_db"
DATA_PATH = "./data"

# === Load Vectorstore ===
def get_vectorstore():
    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        raise Exception("❌ ChromaDB belum ada. Jalankan 'build_dataset.py' dulu untuk membuat dataset.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    print("✅ ChromaDB berhasil dimuat.")
    return vectordb


# === Inisialisasi Chain ===
def create_chain():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.3,
    )

    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt = get_prompt()

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

import re

def format_to_list(text: str) -> str:
    """Ubah format teks menjadi daftar tanpa tabel dan HTML."""
    # Ubah **bold** jadi bold biasa tanpa tag HTML
    # text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # Hapus simbol bullet '•' yang tidak diinginkan
    # text = re.sub(r"•", "", text)

    # Hapus tag <br> dan <br /> yang tidak diperlukan
    text = re.sub(r"<br\s*/?>", "\n", text)

    # Hapus garis atau karakter yang tidak perlu dari format tabel
    # text = re.sub(r"[--]+", "", text)

    # Bersihkan spasi berlebih
    text = re.sub(r"\n{2,}", "\n", text).strip()

    return text




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

        # 🔹 Format teks sebelum dikirim
        formatted_answer = format_to_list(answer)

        save_chatlog(user_text, answer, user_id, 1)

        print(f"💬 Bot menjawab: {formatted_answer}")
        await update.message.reply_text(
            formatted_answer,
            parse_mode="Markdown",
            # disable_web_page_preview=True
        )

        # Simpan konteks terbaru (maksimal 5 percakapan terakhir)
        new_context = f"{previous_context}\nPengguna: {user_text}\nBot: {formatted_answer}"
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
