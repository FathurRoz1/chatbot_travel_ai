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

    # === PENGECEKAN DUPLIKAT (nama file) ===
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



if __name__ == "__main__":
    # Development run
    app.run(host="0.0.0.0", port=8001, debug=True)
