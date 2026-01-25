import os
import re
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv
from chatlog_db import save_chatlog

# === LangChain dan Chroma ===
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
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


def classify_answer_status(answer: str) -> int:
    """
    Return:
      0 = bot tidak bisa menjawab / fallback
      1 = bot bisa menjawab
    """
    if not answer:
        return 0

    a = answer.strip().lower()

    # Frasa yang menandakan "tidak ada info / tidak menemukan"
    not_found_signals = [
        "tidak menemukan informasi",
        "tidak menemukan info",
        "saya tidak menemukan",
        "tidak ada informasi",
        "tidak ada info",
        "tidak tersedia",
        "belum tersedia",
        "saya tidak memiliki informasi",
        "saya tidak punya informasi",
        "saya tidak memiliki data",
        "saya tidak punya data",
        "saya tidak dapat menemukan",
        "saya tidak bisa menemukan",
        "maaf, saya tidak",
        "maaf saya tidak",
    ]

    # Frasa yang biasanya muncul pada jawaban fallback (menyarankan sumber lain)
    redirect_signals = [
        "silakan kunjungi",
        "silahkan kunjungi",
        "website resmi",
        "hubungi kontak",
        "kontak yang tersedia",
        "untuk informasi lebih lanjut",
    ]

    # Frasa yang menandakan "tidak ada di dokumen"
    doc_signals = [
        "dalam dokumen ini",
        "di dokumen ini",
        "pada dokumen ini",
    ]

    score = 0
    if any(p in a for p in not_found_signals):
        score += 2
    if any(p in a for p in redirect_signals):
        score += 1
    if any(p in a for p in doc_signals):
        score += 1

    # Bonus: pola umum "Maaf, ... tidak ... (informasi/data)"
    if re.search(r"maaf[, ]+.*tidak.*(informasi|data)", a):
        score += 1

    # Ambang batas: 2 atau lebih dianggap "tidak bisa menjawab"
    return 0 if score >= 2 else 1

# === Handler Pesan User ===
# Memory sederhana untuk menyimpan konteks tiap user
user_memory = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    user_id = update.effective_chat.id

    print(f"📩 Pesan diterima dari {user_id}: {user_text}")
    await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)

    previous_context = user_memory.get(user_id, "")

    try:
        full_input = f"{previous_context}\n\nPengguna: {user_text}"

        response = chain.invoke(full_input)
        answer = response.content.strip()

        formatted_answer = format_to_list(answer)

        # ✅ Tentukan status dari jawaban bot
        status = classify_answer_status(answer)

        # ✅ Simpan dengan status yang benar
        save_chatlog(user_text, formatted_answer, user_id, status)

        print(f"💬 Bot menjawab: {formatted_answer}")
        await update.message.reply_text(
            formatted_answer,
            parse_mode="Markdown",
        )

        new_context = f"{previous_context}\nPengguna: {user_text}\nBot: {formatted_answer}"
        user_memory[user_id] = "\n".join(new_context.splitlines()[-10:])

    except Exception as e:
        print(f"❌ Error: {e}")
        # optional: simpan error sebagai status 0 atau status khusus (mis. -1)
        save_chatlog(user_text, f"ERROR: {e}", user_id, 0)

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
