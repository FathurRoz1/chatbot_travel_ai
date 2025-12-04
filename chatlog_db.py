import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

# === Load .env file ===
load_dotenv()

# === Koneksi Database PostgreSQL ===
def get_db_connection():
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            dbname=os.getenv("DB_NAME", "your_database_name"),
            user=os.getenv("DB_USER", "your_user"),
            password=os.getenv("DB_PASSWORD", "your_password"),
        )
        return connection
    except Exception as e:
        print(f"❌ Koneksi ke database gagal: {e}")
        return None

# === Simpan Chat Log ke Database ===
def save_chatlog(question: str, answer: str, api_key: str, status: int):
    connection = get_db_connection()
    if connection is None:
        return

    cursor = connection.cursor()
    
    try:
        query = sql.SQL("""
            INSERT INTO public.h_chatlog (question, answer, api_key, status)
            VALUES (%s, %s, %s, %s)
        """)
        
        cursor.execute(query, (question, answer, api_key, status))
        connection.commit()
        print("✅ Chatlog berhasil disimpan ke database.")
    except Exception as e:
        print(f"❌ Gagal menyimpan chatlog: {e}")
    finally:
        cursor.close()
        connection.close()

