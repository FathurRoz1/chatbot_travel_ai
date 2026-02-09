import os
import sys
import re
import subprocess
import json
from pathlib import Path
from filelock import FileLock, Timeout
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==== Konfigurasi ====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
BUILD_SCRIPT = BASE_DIR / "build_dataset.py"
LOCK_FILE = BASE_DIR / ".build.lock"

API_TOKEN = os.getenv("DATASET_API_TOKEN", "CHANGE_ME")

# Batasi ukuran upload (contoh 50MB)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

DATA_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_FILE = BASE_DIR / "processed_files.json"

def _load_processed_files() -> set:
    if PROCESSED_FILE.exists():
        try:
            return set(json.loads(PROCESSED_FILE.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()



def _auth_or_401():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth.split(" ", 1)[1].strip()
    return token == API_TOKEN


def _safe_filename(name: str) -> str:
    name = os.path.basename(name or "uploaded.bin")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)  # aman untuk filesystem
    return name


def _unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, suf = dest.stem, dest.suffix
    i = 1
    while True:
        cand = dest.parent / f"{stem}_{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def _run_build() -> dict:
    """
    Menjalankan build_dataset.py (incremental karena ada processed_files.json).
    """
    if not BUILD_SCRIPT.exists():
        raise FileNotFoundError(f"build script not found: {BUILD_SCRIPT}")

    # Jalankan pakai python env yang sama
    proc = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout[-8000:],  # potong biar response tidak kebesaran
        "stderr": proc.stderr[-8000:],
    }


def _save_processed_files(processed: set) -> None:
    PROCESSED_FILE.write_text(
        json.dumps(sorted(list(processed)), ensure_ascii=False),
        encoding="utf-8"
    )


def _delete_from_chroma(filename: str) -> dict:
    chroma_dir = BASE_DIR / "chroma_db"

    # lazy import supaya flask tetap ringan kalau endpoint ini tidak dipakai
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
    except Exception as e:
        return {"ok": False, "error": f"Chroma/LangChain import failed: {e}"}

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)

    rel_source = str(Path("data") / filename)
    abs_source = str((DATA_DIR / filename).resolve())

    # Variasi path untuk jaga-jaga (Linux/Windows)
    candidates = list(dict.fromkeys([
        rel_source,
        rel_source.replace("/", "\\"),
        abs_source,
        abs_source.replace("/", "\\"),
    ]))

    errors = []
    # Metadata baru (jika build_dataset.py sudah ditambah)
    try:
        db.delete(where={"dataset_file": filename})
    except Exception as e:
        errors.append(f"delete(where=dataset_file) failed: {e}")

    # Metadata bawaan loader (source)
    for src in candidates:
        try:
            db.delete(where={"source": src})
        except Exception as e:
            errors.append(f"delete(where=source={src}) failed: {e}")

    # Persist jika method tersedia
    try:
        if hasattr(db, "persist"):
            db.persist()
    except Exception as e:
        errors.append(f"persist failed: {e}")

    return {"ok": True, "attempted_sources": candidates, "errors": errors}


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/datasets/upload")
def upload_dataset():
    if not _auth_or_401():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"ok": False, "error": "file is required"}), 400

    f = request.files["file"]
    filename = _safe_filename(f.filename)

    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".txt")):
        return jsonify({"ok": False, "error": "Only .pdf or .txt allowed"}), 400

    # === PENGECEKAN DUPLIKAT 
    processed = _load_processed_files()
    file_path = DATA_DIR / filename

    if file_path.exists() or (filename in processed):
        return jsonify({
            "ok": False,
            "message": "file sudah ada",
            "filename": filename,
            "exists_in_data_dir": file_path.exists(),
            "already_built": (filename in processed),
        }), 409

    # jika lolos cek, baru simpan & build
    f.save(file_path)

    lock = FileLock(str(LOCK_FILE))
    try:
        with lock.acquire(timeout=300):
            build_result = _run_build()
    except Timeout:
        return jsonify({"ok": False, "error": "Build is busy (lock timeout). Try again."}), 429
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "file_saved": filename}), 500

    status = 200 if build_result["returncode"] == 0 else 500
    return jsonify({
        "ok": build_result["returncode"] == 0,
        "message": "uploaded & build executed",
        "saved_as": filename,
        "build": build_result
    }), status


@app.post("/datasets/delete")
def delete_dataset():
    if not _auth_or_401():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    filename = payload.get("filename") or request.args.get("filename") or request.form.get("filename")
    if not filename:
        return jsonify({"ok": False, "error": "filename is required"}), 400

    filename = _safe_filename(filename)
    file_path = DATA_DIR / filename

    lock = FileLock(str(LOCK_FILE))
    try:
        with lock.acquire(timeout=300):
            # 1) hapus file fisik (kalau ada)
            file_deleted = False
            if file_path.exists():
                try:
                    file_path.unlink()
                    file_deleted = True
                except Exception as e:
                    return jsonify({"ok": False, "error": f"failed to delete file: {e}", "filename": filename}), 500

            # 2) hapus dari processed_files.json supaya bisa upload ulang dengan nama sama
            processed = _load_processed_files()
            was_in_processed = filename in processed
            if was_in_processed:
                processed.discard(filename)
                _save_processed_files(processed)

            # 3) hapus dari ChromaDB berdasarkan metadata source / dataset_file
            chroma_result = _delete_from_chroma(filename)

    except Timeout:
        return jsonify({"ok": False, "error": "Build is busy (lock timeout). Try again."}), 429
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    # Jika tidak ada apa-apa yang bisa dihapus, berikan 404
    if (not file_deleted) and (not was_in_processed):
        return jsonify({
            "ok": False,
            "message": "dataset tidak ditemukan (file tidak ada dan tidak tercatat di processed_files.json)",
            "filename": filename,
            "chroma": chroma_result
        }), 404

    return jsonify({
        "ok": True,
        "message": "dataset deleted",
        "filename": filename,
        "file_deleted": file_deleted,
        "was_in_processed": was_in_processed,
        "chroma": chroma_result
    }), 200





if __name__ == "__main__":
    # Development run
    app.run(host="0.0.0.0", port=8001, debug=True)
