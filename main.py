import requests
import logging
import os
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# === Konfigurasi logging ke console ===
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# === Ganti dengan token dan API key kamu ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-safeguard-20b")


def ask_groq(prompt: str):
    """Kirim prompt ke API Groq dan ambil balasan"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Kamu adalah asisten AI yang ramah dan menjawab dengan bahasa Indonesia."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    else:
        return f"❌ Terjadi kesalahan dari API: {data}"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logger.info(f"User {user.first_name} ({user.id}) menggunakan /start")
    await update.message.reply_text("Halo 👋! Saya Chatbot Institute Rotten Nation. Silakan kirim pertanyaan apa pun!")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_text = update.message.text

    logger.info(f"📩 Pesan dari {user.first_name} ({user.id}): {user_text}")

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        answer = ask_groq(user_text)
        await update.message.reply_text(answer)
        logger.info(f"💬 Bot membalas ke {user.first_name}: {answer}")
    except Exception as e:
        error_msg = f"⚠️ Terjadi error: {e}"
        await update.message.reply_text(error_msg)
        logger.error(error_msg)


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot sedang berjalan...")
    app.run_polling()


if __name__ == "__main__":
    main()
