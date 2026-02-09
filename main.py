import os
import re
import asyncio
import traceback
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from chatlog_db import save_chatlog

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from prompt_template import get_prompt


# ============================================================
# ENV
# ============================================================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

# Project kamu (sesuai screenshot) => semua ada 1 folder yang sama
BASE_DIR = Path(__file__).resolve().parent

# ✅ PENTING: ini harus sama persis dengan folder kamu: "chroma_db"
CHROMA_DIR = BASE_DIR / "chroma_db"
PROCESSED_FILE = BASE_DIR / "processed_files.json"
VERSION_FILE = BASE_DIR / ".dataset_version"  # opsional (kalau ada)

_reload_lock = asyncio.Lock()
_last_sig_ns = 0

vectordb = None
chain = None


# ============================================================
# Heavy init
# ============================================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0.3,
)


# ============================================================
# Dataset signature (anti restart)
# ============================================================
def _mtime_ns(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


def _chroma_mtime_ns() -> int:
    if not CHROMA_DIR.exists():
        return 0

    # paling umum ada file sqlite
    sqlite_path = CHROMA_DIR / "chroma.sqlite3"
    if sqlite_path.exists():
        return _mtime_ns(sqlite_path)

    # fallback scan
    sig = 0
    for f in CHROMA_DIR.rglob("*"):
        if f.is_file():
            sig = max(sig, _mtime_ns(f))
    return sig


def dataset_signature_ns() -> int:
    sig = 0
    sig = max(sig, _mtime_ns(VERSION_FILE))
    sig = max(sig, _mtime_ns(PROCESSED_FILE))
    sig = max(sig, _chroma_mtime_ns())
    return sig


def _build_chain(vdb: Chroma):
    retriever = vdb.as_retriever(search_kwargs={"k": 10})
    prompt = get_prompt()
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )


async def ensure_chain_latest(force: bool = False):
    global vectordb, chain, _last_sig_ns

    sig = dataset_signature_ns()
    need_reload = force or (chain is None) or (sig > _last_sig_ns)
    if not need_reload:
        return

    async with _reload_lock:
        sig2 = dataset_signature_ns()
        need_reload2 = force or (chain is None) or (sig2 > _last_sig_ns)
        if not need_reload2:
            return

        if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
            raise Exception(f"❌ ChromaDB tidak ditemukan di: {CHROMA_DIR}")

        try:
            new_vdb = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
            new_chain = _build_chain(new_vdb)

            vectordb = new_vdb
            chain = new_chain
            _last_sig_ns = sig2

            print(f"🔄 Reload OK | sig={sig2}")
        except Exception as e:
            print("❌ Reload gagal (pakai chain lama jika ada):", e)
            traceback.print_exc()
            if chain is None:
                raise


# ============================================================
# Formatting + status classify
# ============================================================
def format_to_list(text: str) -> str:
    """Ubah format teks menjadi daftar tanpa tabel dan HTML."""
    # Ubah **bold** jadi bold biasa tanpa tag HTML
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # Hapus simbol bullet '•' yang tidak diinginkan
    # text = re.sub(r"•", "", text)

    # Hapus tag <br> dan <br /> yang tidak diperlukan
    text = re.sub(r"<br\s*/?>", "\n", text)

    # Hapus garis atau karakter yang tidak perlu dari format tabel
    text = re.sub(r"[--]+", "", text)

    # Bersihkan spasi berlebih
    text = re.sub(r"\n{2,}", "\n", text).strip()

    return text


def classify_answer_status(answer: str) -> int:
    if not answer:
        return 0

    a = answer.strip().lower()

    not_found_signals = [
        "tidak menemukan informasi", "tidak menemukan info", "saya tidak menemukan",
        "tidak ada informasi", "tidak ada info", "tidak tersedia", "belum tersedia",
        "saya tidak memiliki informasi", "saya tidak punya informasi",
        "saya tidak memiliki data", "saya tidak punya data",
        "saya tidak dapat menemukan", "saya tidak bisa menemukan",
        "maaf, saya tidak", "maaf saya tidak",
    ]
    redirect_signals = [
        "silakan kunjungi", "silahkan kunjungi", "website resmi",
        "hubungi kontak", "kontak yang tersedia", "untuk informasi lebih lanjut",
    ]
    doc_signals = ["dalam dokumen ini", "di dokumen ini", "pada dokumen ini"]
    answer_signals = [
        "harga", "rp", "kapasitas", "include", "exclude",
        "paket", "sewa", "tersedia", "fasilitas"
    ]

    if any(p in a for p in answer_signals):
        return 1
    
    score = 0
    if any(p in a for p in not_found_signals):
        score += 2
    if any(p in a for p in redirect_signals):
        score += 1
    if any(p in a for p in doc_signals):
        score += 1
    if re.search(r"maaf[, ]+.*tidak.*(informasi|data)", a):
        score += 1

    return 0 if score >= 2 else 1


# ============================================================
# Telegram handlers
# ============================================================
user_memory = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Halo 👋! Saya Chatbot Virtual Assistant Travel Malang ID. Silahkan tanyakan apa saja seputar travel di Malang."
    )


async def debug_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"BASE_DIR: {BASE_DIR}\n"
        f"CHROMA_DIR: {CHROMA_DIR} (exists={CHROMA_DIR.exists()})\n"
        f"SQLITE: {CHROMA_DIR / 'chroma.sqlite3'} (exists={(CHROMA_DIR / 'chroma.sqlite3').exists()})\n"
        f"PROCESSED_FILE: {PROCESSED_FILE} (exists={PROCESSED_FILE.exists()})\n"
        f"VERSION_FILE: {VERSION_FILE} (exists={VERSION_FILE.exists()})\n"
        f"SIG_NS: {dataset_signature_ns()}\n"
        f"LAST_SIG_NS: {_last_sig_ns}\n"
        f"CHAIN_READY: {chain is not None}\n"
    )
    await update.message.reply_text(f"```{msg}```", parse_mode="Markdown")


async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await ensure_chain_latest(force=True)
        await update.message.reply_text("✅ Reload dataset berhasil (tanpa restart).")
    except Exception as e:
        await update.message.reply_text(f"❌ Gagal reload: {e}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = (update.message.text or "").strip()
    user_id = update.effective_chat.id

    print(f"📩 Pesan diterima dari {user_id}: {user_text}")
    await context.bot.send_chat_action(chat_id=user_id, action=ChatAction.TYPING)

    previous_context = user_memory.get(user_id, "")

    try:
        await ensure_chain_latest()

        full_input = f"{previous_context}\n\nPengguna: {user_text}"
        response = chain.invoke(full_input)
        answer = (response.content or "").strip()

        formatted_answer = format_to_list(answer)

        status = classify_answer_status(answer)
        save_chatlog(user_text, answer, user_id, status)

        await update.message.reply_text(formatted_answer, parse_mode="Markdown")

        new_context = f"{previous_context}\nPengguna: {user_text}\nBot: {formatted_answer}"
        user_memory[user_id] = "\n".join(new_context.splitlines()[-10:])

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        save_chatlog(user_text, f"ERROR: {e}", user_id, 0)
        await update.message.reply_text("⚠️ Maaf, terjadi kesalahan saat memproses pesan Anda.")


def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN belum di-set di .env")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY belum di-set di .env")

    print("🤖 Bot berjalan...")
    print("BASE_DIR =", BASE_DIR)
    print("CHROMA_DIR =", CHROMA_DIR)

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("debug", debug_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
