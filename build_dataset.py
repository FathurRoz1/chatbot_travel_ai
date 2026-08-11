import os
import gc
import json

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
PROCESSED_FILE = "processed_files.json"
BATCH_SIZE = 4  # Jumlah chunk per batch embedding (hemat RAM)

# Membaca daftar file yang sudah diproses sebelumnya
if os.path.exists(PROCESSED_FILE):
    with open(PROCESSED_FILE, "r") as f:
        processed_files = set(json.load(f))
else:
    processed_files = set()


def load_documents():
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

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
        loaded = loader.load()
        # Tambahkan metadata agar bisa dihapus per-file dari Chroma
        for d in loaded:
            try:
                d.metadata["dataset_file"] = file
            except Exception:
                pass
        docs.extend(loaded)

        # Tambahkan ke daftar file yang sudah diproses
        processed_files.add(file)

    return docs


def build_dataset():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # === Tahap 1: Muat dokumen ===
    print("[INFO] Memuat dokumen...")
    documents = load_documents()
    print(f"[OK] Ditemukan {len(documents)} dokumen baru.")

    if len(documents) == 0:
        print("[OK] Tidak ada file baru untuk diproses.")
        return

    # === Tahap 2: Split dokumen ===
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)
    print(f"[INFO] Terbagi menjadi {len(texts)} potongan teks.")

    # Bebaskan memori dokumen asli (tidak diperlukan lagi)
    del documents
    del splitter
    gc.collect()

    # === Tahap 3: Muat embedding model (lazy load) ===
    print("[INFO] Memuat model embedding (ini mungkin butuh waktu)...")
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": BATCH_SIZE, "normalize_embeddings": False},
    )
    print("[OK] Model embedding berhasil dimuat.")

    # === Tahap 4: Simpan ke ChromaDB dalam batch ===
    from langchain_chroma import Chroma

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"[INFO] Menyimpan batch {batch_num}/{total_batches} ({len(batch)} chunk)...")
        db.add_documents(batch)
        gc.collect()  # Bebaskan memori setiap batch

    # === Tahap 5: Simpan daftar file yang sudah diproses ===
    with open(PROCESSED_FILE, "w") as f:
        json.dump(list(processed_files), f)

    print(f"[OK] Dataset berhasil diperbarui. Total {total} chunk disimpan.")


if __name__ == "__main__":
    build_dataset()
    