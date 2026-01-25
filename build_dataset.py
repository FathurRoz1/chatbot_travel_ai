from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os, json

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
PROCESSED_FILE = "processed_files.json"

# Inisialisasi model embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Membaca daftar file yang sudah diproses sebelumnya
if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r") as f:
        processed_files = set(json.load(f))
else:
    processed_files = set()

def load_documents():
    docs = []
    for file in os.listdir(DATA_DIR):
        # Lewati file yang sudah pernah diproses
        if file in processed_files:
            print(f"[INFO] Lewati {file} (sudah pernah diproses)")
            continue

        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            print(f"[ERROR] Format file tidak didukung: {file}")
            continue

        print(f"[INFO] Memuat {file} ...")
        docs.extend(loader.load())

        # Tambahkan ke daftar file yang sudah diproses
        processed_files.add(file)

    return docs

print("[INFO] Memuat dokumen...")
documents = load_documents()
print(f"[OK] Ditemukan {len(documents)} dokumen baru.")

if len(documents) > 0:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)
    print(f"[INFO] Terbagi menjadi {len(texts)} potongan teks.")

    print("[INFO] Menyimpan ke ChromaDB...")
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    db.add_documents(texts)
    # db.persist()

    # Simpan daftar file yang sudah diproses
    with open(PROCESSED_FILE, "w") as f:
        json.dump(list(processed_files), f)

    print("[OK] Dataset berhasil diperbarui tanpa duplikasi.")
else:
    print("[OK] Tidak ada file baru untuk diproses.")