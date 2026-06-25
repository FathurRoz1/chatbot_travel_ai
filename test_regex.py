import re

def test_regex():
    text = """Berikut beberapa paket wisata:
1. Bromo Midnight Private Family
- Durasi: 1 hari 1 malam
- Target: Keluarga
- Fasilitas:
  • Transportasi pribadi
  • Driver
2. Sewa Jeep Only Bromo Sunrise
- Durasi: Satu hari
- Target: Peserta bawa kendaraan sendiri
3. Muslim Friendly Malang Batu
- Durasi: 2 hari 1 malam
"""

    text2 = re.sub(r"(?<!\n)\n(\d+\.\s)", r"\n\n\1", text)
    print("=== AFTER REGEX ===")
    print(text2)

if __name__ == "__main__":
    test_regex()
