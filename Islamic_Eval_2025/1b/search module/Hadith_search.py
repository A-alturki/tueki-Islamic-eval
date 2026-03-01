import json
import numpy as np
from pyarabic.araby import strip_tashkeel, strip_tatweel
from pyarabic.normalize import normalize_lamalef
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict


class HadithSearchEngine:
    def __init__(self):
        self.hadith_data = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.hadith_texts = []
        self.hadith_ids = []

    def preprocess_arabic_text(self, text):
        """Basic Arabic text preprocessing"""
        if not text:
            return ""

        # Remove diacritics (tashkeel)
        arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        text = arabic_diacritics.sub('', text)

        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Remove curly braces and other formatting
        text = re.sub(r'[{}]', '', text)
        text = re.sub(r'\r\n|\n|\r', ' ', text)

        text = strip_tashkeel(text)
        text = strip_tatweel(text)
        text = normalize_lamalef(text)

        return text

    def load_data_from_file(self, json_file_path):
        """Load hadith data from JSON file"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                self.hadith_data = json.load(file)

            print(f"Successfully loaded {len(self.hadith_data)} hadith records from {json_file_path}")

            # Extract texts for indexing
            for hadith in self.hadith_data:
                hadith_id = hadith.get('hadithID')

                # Combine all searchable text fields
                searchable_text = ""

                if hadith.get('hadithTxt'):
                    searchable_text += self.preprocess_arabic_text(hadith['hadithTxt'])

                if hadith.get('Matn'):
                    searchable_text += " " + self.preprocess_arabic_text(hadith['Matn'])

                if hadith.get('title'):
                    searchable_text += " " + self.preprocess_arabic_text(hadith['title'])

                self.hadith_texts.append(searchable_text)
                self.hadith_ids.append(hadith_id)

        except FileNotFoundError:
            print(f"Error: File '{json_file_path}' not found.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in '{json_file_path}': {e}")
            raise
        except Exception as e:
            print(f"Error loading file '{json_file_path}': {e}")
            raise

    def load_data(self, hadith_json_data):
        """Load hadith data from JSON data (keeping for backward compatibility)"""
        self.hadith_data = hadith_json_data

        # Extract texts for indexing
        for hadith in self.hadith_data:
            hadith_id = hadith.get('hadithID')

            # Combine all searchable text fields
            searchable_text = ""

            if hadith.get('hadithTxt'):
                searchable_text += self.preprocess_arabic_text(hadith['hadithTxt'])

            if hadith.get('Matn'):
                searchable_text += " " + self.preprocess_arabic_text(hadith['Matn'])

            if hadith.get('title'):
                searchable_text += " " + self.preprocess_arabic_text(hadith['title'])

            self.hadith_texts.append(searchable_text)
            self.hadith_ids.append(hadith_id)

    def build_index(self):
        """Build TF-IDF index"""
        if not self.hadith_data:
            raise ValueError("No hadith data loaded. Please load data first using load_data_from_file() or load_data().")

        print("Building TF-IDF index...")

        # Configure TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=4000,  # Limit vocabulary size
            stop_words=None,  # No Arabic stopwords in sklearn
            ngram_range=(1, 7),  # Use unigrams and bigrams
            min_df=1,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            sublinear_tf=True,  # Apply sublinear scaling
            analyzer='char'
        )

        # Fit and transform the hadith texts
        self.tfidf_matrix = self.vectorizer.fit_transform(self.hadith_texts)

        print(f"Index built successfully!")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"Number of documents: {self.tfidf_matrix.shape[0]}")

    def search(self, query, top_k=10):
        """Search for similar hadith based on query"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Preprocess query
        processed_query = self.preprocess_arabic_text(query)

        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([processed_query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                # Handle hadithTxt - check if it's None before processing
                hadith_txt = self.hadith_data[idx].get('hadithTxt', '')
                if hadith_txt is None:
                    hadith_txt = ''
                hadith_txt_display = hadith_txt

                # Handle Matn - check if it's None before processing
                matn = self.hadith_data[idx].get('Matn', '')
                if matn is None:
                    matn = ''
                matn_display = matn

                results.append({
                    'hadithID': self.hadith_ids[idx],
                    'similarity_score': float(similarities[idx]),
                    'title': self.hadith_data[idx].get('title', ''),
                    'hadithTxt': hadith_txt_display,
                    'Matn': matn_display
                })

        return results


# Initialize the search engine
search_engine = HadithSearchEngine()

# Load data from JSON file - REPLACE 'hadith_data.json' with your actual file path
json_file_path = 'resources/six_hadith_books/six_hadith_books.json'  # Update this with your JSON file path

search_engine.load_data_from_file(json_file_path)
search_engine.build_index()

# Test with sample queries
sample_queries = [
    "ألا وإن في الجسد مضغة إذا صلحت صلح الجسد كله، وإذا فسدت فسد الجسد كله، ألا وهي القل",
    " ا تحاسدوا، ولا تناجشوا، ولا تباغضوا، ولا تدابروا، ولا يبع بعضكم على بيع بعض، وكونوا عباد الله إخوانا، المسلم أخو المسلم، لا يظلمه، ولا يخذله، ولا يحقره، التقوى ها هنا - ويشير إلى صدره ثلاث مرات - بحسب امرئ من الشر أن يحقر أخاه المسلم، كل المسلم على المسلم حرام: دمه، وماله، وعرضه",
    "إنما النساء شقائق الرجال",
    "إن الله لا ينظر إلى صوركم وأموالكم ولكن ينظر إلى قلوبكم وأعمالكم",
    "إن الله جميل يحب الجمال"
]

print("\n" + "=" * 80)
print("HADITH SEARCH RESULTS")
print("=" * 80)

for query in sample_queries:
    print(f"\n🔍 Search Query: '{query}'")
    print("-" * 60)

    results = search_engine.search(query, top_k=10)

    if not results:
        print("No results found.")
        continue

    for i, result in enumerate(results, 1):  # Show top 5 for brevity
        print(f"{i}. Hadith ID: {result['hadithID']}")
        print(f"   Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Title: {result['title']}")
        if result['Matn']:
            print(f"   Matn: {result['Matn']}")
        print()
