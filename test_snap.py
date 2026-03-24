import sys, io, re, unicodedata, json
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
from typing import Optional, List
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

QURAN_FILE  = Path("datasets/quranic_verses.json")
HADITH_FILE = Path("datasets/six_hadith_books.json")
OUT_DIR     = Path("results_rescored")

# paste just the functions needed
exec(open("rescore_1c.py", encoding="utf-8").read().split("# ── Main rescoring loop")[0])

model_raw = "وَمَن يُقَاتِلْ فِي سَبِيلِ اللَّهِ فَيُقْتَلْ أَوْ يَغْلِبْ فَسَوْفَ نُؤْتِيهِ أَجْرًا عَظِيمًا (74)"
gold      = "وَمَن يُقَاتِلْ فِي سَبِيلِ اللَّهِ فَيُقْتَلْ أَوْ يَغْلِبْ فَسَوْفَ نُؤْتِيهِ أَجْرًا عَظِيمًا"

result = snap(model_raw, "Ayah")
print("snapped:", result)
print("gold:   ", gold)
print("match:  ", remove_default_diac(result) == remove_default_diac(gold))
