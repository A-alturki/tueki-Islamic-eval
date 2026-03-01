# Data Analysis Notes

## Dev Ground Truth (dev_SubtaskC.tsv) - 180 rows
- 50 Questions (C-Q01 to C-Q50)
- ~70 WrongAyah, ~110 WrongHadith labels

### Correction distribution:
- خطأ (no valid correction): 111/180 = 61.7%
- Valid corrections: 69/180 = 38.3%

### Key observations about خطأ cases:
- Fabricated hadiths that don't match anything in the 6 books → legitimately خطأ
- Very short or garbled spans → legitimately خطأ
- Some ayah spans are purely hallucinated (don't exist in Quran) → legitimately خطأ

### Key observations about valid correction cases:
- WrongAyah: The LLM cited a wrong ayah but correct text exists → correction = canonical text
- WrongHadith: The LLM cited an actual hadith with slight variation → correction = exact Matn/hadithTxt
- Many corrections are multi-ayah (span multiple consecutive ayahs with numbers in parens)
- Cross-ayah cases: "(18) وَاقْصِدْ فِي مَشْيِكَ..." style, combining ayahs

## Test Data (Test_Subtask_1C_USER.tsv) - 308 rows
- 60+ questions (C-Q001 onwards)
- Mix of Ayah and Hadith spans
- HUMAIN's submission: 59/308 corrected (19.2%), 249/308 خطأ (80.8%)
- This ratio seems wrong given dev set has 38.3% corrections
- Suggests system is over-predicting خطأ on test set

## Correction Format Examples
### Single Quran ayah:
`وَلَا تُصَعِّرْ خَدَّكَ لِلنَّاسِ وَلَا تَمْشِ فِي الْأَرْضِ مَرَحًا ۖ إِنَّ اللَّهَ لَا يُحِبُّ كُلَّ مُخْتَالٍ فَخُورٍ`

### Multi-ayah (with verse numbers in parens):
`وَلَا تُصَعِّرْ خَدَّكَ...(18) وَاقْصِدْ فِي مَشْيِكَ...(19)`

### Hadith (Matn or full text):
`من قُتل دونَ أهلِه فهو شهيدٌ، ومن قُتل دون دينه فهو شهيدٌ`

## Error Types in LLM Responses
1. **Wrong ayah cited**: LLM names a correct-sounding surah/ayah but gives different text
2. **Wrong diacritization**: Same text but with incorrect harakat
3. **Hallucinated Hadith**: Invented text that sounds authentic but isn't in any book
4. **Partial/truncated**: Only part of a verse cited
5. **Mixed/confused**: Parts from different ayahs blended together
6. **Wrong numbering**: LLM cites "ayah 18-19" but the span contains different text

## Reference Database
### six_hadith_books.json
- Bukhari, Muslim, Abu Dawud, Tirmidhi, Ibn Majah, Nasai
- ~34,994 hadiths total, 31,811 with Matn field
- Both full text (isnad+matn) and matn-only indexed

### quranic_verses.json
- All 6,236 ayahs of the Quran
- Full text with tashkeel (diacritics)
- 114 surahs
