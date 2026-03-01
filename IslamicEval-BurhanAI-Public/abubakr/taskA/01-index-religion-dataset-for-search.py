#!/usr/bin/env python3
"""
Religious Text Search Index Builder
Builds optimized search indices for Quran and Hadith datasets for fast verification.
"""

import json
import pandas as pd
import pickle
from pathlib import Path
import re
from collections import defaultdict
import hashlib
import argparse
from tqdm import tqdm

try:
    import pyarabic.araby as araby
except ImportError:
    print("Warning: pyarabic not installed. Install with: pip install pyarabic")
    # Fallback normalization
    def normalize_arabic(text):
        return re.sub(r'[^\u0600-\u06FF\s]', '', text).strip()
    araby = type('obj', (object,), {'strip_diacritics': normalize_arabic, 'normalize_hamza': lambda x: x})()

try:
    from whoosh.index import create_in, open_dir
    from whoosh.fields import Schema, TEXT, ID, NUMERIC
    from whoosh.analysis import StandardAnalyzer
    from whoosh.qparser import QueryParser
    from whoosh import scoring
    WHOOSH_AVAILABLE = True
except ImportError:
    print("Warning: whoosh not installed. Install with: pip install whoosh")
    WHOOSH_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    print("Warning: fuzzywuzzy not installed. Install with: pip install fuzzywuzzy")
    FUZZYWUZZY_AVAILABLE = False

try:
    import cohere
    from dotenv import dotenv_values
    COHERE_AVAILABLE = True
except ImportError:
    print("Warning: cohere or python-dotenv not installed. Install with: pip install cohere python-dotenv")
    COHERE_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
    QDRANT_AVAILABLE = True
except ImportError:
    print("Warning: qdrant-client not installed. Install with: pip install qdrant-client")
    QDRANT_AVAILABLE = False

class ReligiousTextIndexer:
    def __init__(self):
        self.quran_index = {
            'exact': {},           # hash -> ayah_data
            'normalized': {},      # normalized_text -> ayah_data
            'length_buckets': defaultdict(list),  # length -> [ayah_data]
            'ngrams': defaultdict(set),           # 3gram -> {ayah_ids}
            'fuzzy_texts': [],     # For fuzzywuzzy processing
        }
        
        self.hadith_index = {
            'exact': {},
            'normalized': {},
            'length_buckets': defaultdict(list),
            'ngrams': defaultdict(set),
            'fuzzy_texts': [],     # For fuzzywuzzy processing
        }
        
        # Whoosh search indices
        self.quran_whoosh_index = None
        self.hadith_whoosh_index = None
        
        # Embedding and vector search
        self.cohere_client = None
        self.qdrant_client = None
        self.embedding_dimension = 1536  # Cohere embed-v4.0 dimension
        
        # Initialize Cohere client
        if COHERE_AVAILABLE:
            try:
                config = dotenv_values("../.env") 
                if config.get("COHERE_API_KEY"):
                    self.cohere_client = cohere.ClientV2(api_key=config.get("COHERE_API_KEY"))
            except Exception as e:
                print(f"Warning: Could not initialize Cohere client: {e}")
        
        # Initialize Qdrant client
        if QDRANT_AVAILABLE:
            try:
                self.qdrant_client = QdrantClient(path="../datasets/indices/qdrant_db")
            except Exception as e:
                print(f"Warning: Could not initialize Qdrant client: {e}")
    
    def create_whoosh_schema(self):
        """Create Whoosh schema for full-text search"""
        if not WHOOSH_AVAILABLE:
            return None
        
        from whoosh.fields import Schema, TEXT, ID, NUMERIC
        from whoosh.analysis import StandardAnalyzer
        
        return Schema(
            id=ID(stored=True, unique=True),
            original_text=TEXT(stored=True, analyzer=StandardAnalyzer()),
            normalized_text=TEXT(stored=True, analyzer=StandardAnalyzer()),
            text_length=NUMERIC(stored=True),
            type=ID(stored=True)  # 'ayah' or 'hadith'
        )
    
    def normalize_text(self, text):
        """Comprehensive Arabic text normalization"""
        if not text:
            return ""
        
        # Remove diacritics and normalize
        normalized = araby.strip_diacritics(text)
        normalized = araby.normalize_hamza(normalized)
        
        # Clean whitespace and punctuation
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\u0600-\u06FF\s]', '', normalized)
        
        return normalized.strip()
    
    def get_text_hash(self, text):
        """Generate hash for exact matching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def generate_ngrams(self, text, n=3):
        """Generate character n-grams for fuzzy matching"""
        text = text.replace(' ', '')
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    def generate_embeddings_batch(self, texts, batch_size=50):
        """Generate embeddings for a batch of texts using Cohere"""
        if not self.cohere_client or not texts:
            return []
        
        try:
            text_inputs = [{"content": [{"type": "text", "text": text}]} for text in texts]
            response = self.cohere_client.embed(
                inputs=text_inputs,
                model="embed-v4.0",
                input_type="search_document",
                embedding_types=["float"],
            )
            return response.embeddings.float
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [None] * len(texts)
    
    def setup_qdrant_collections(self, reset=False):
        """Setup Qdrant collections for Quran and Hadith"""
        if not self.qdrant_client:
            return False
        
        try:
            # Handle Quran collection
            if reset:
                try:
                    self.qdrant_client.delete_collection("quran_embeddings")
                    print("✓ Deleted existing Quran embeddings collection")
                except:
                    pass  # Collection didn't exist
            
            try:
                self.qdrant_client.get_collection("quran_embeddings")
                if not reset:
                    print("✓ Quran embeddings collection already exists")
                else:
                    # Create after reset
                    self.qdrant_client.create_collection(
                        collection_name="quran_embeddings",
                        vectors_config=VectorParams(
                            size=self.embedding_dimension,
                            distance=Distance.COSINE
                        )
                    )
                    print("✓ Created new Quran embeddings collection")
            except:
                self.qdrant_client.create_collection(
                    collection_name="quran_embeddings",
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                print("✓ Created Quran embeddings collection")
            
            # Handle Hadith collection
            if reset:
                try:
                    self.qdrant_client.delete_collection("hadith_embeddings")
                    print("✓ Deleted existing Hadith embeddings collection")
                except:
                    pass  # Collection didn't exist
            
            try:
                self.qdrant_client.get_collection("hadith_embeddings")
                if not reset:
                    print("✓ Hadith embeddings collection already exists")
                else:
                    # Create after reset
                    self.qdrant_client.create_collection(
                        collection_name="hadith_embeddings",
                        vectors_config=VectorParams(
                            size=self.embedding_dimension,
                            distance=Distance.COSINE
                        )
                    )
                    print("✓ Created new Hadith embeddings collection")
            except:
                self.qdrant_client.create_collection(
                    collection_name="hadith_embeddings",
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                print("✓ Created Hadith embeddings collection")
            
            return True
        except Exception as e:
            print(f"Error setting up Qdrant collections: {e}")
            return False
    
    def build_quran_index(self, quran_path="../datasets/raw-data/quranic_verses.json"):
        """Build search index for Quran verses"""
        print("Building Quran index...")
        
        # Initialize Whoosh index
        if WHOOSH_AVAILABLE:
            schema = self.create_whoosh_schema()
            quran_index_dir = Path("../datasets/indices/quran_whoosh")
            quran_index_dir.mkdir(parents=True, exist_ok=True)
            self.quran_whoosh_index = create_in(str(quran_index_dir), schema)
            quran_writer = self.quran_whoosh_index.writer()
        
        with open(quran_path, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        
        # Add progress bar for Quran indexing
        for verse in tqdm(quran_data, desc="Indexing Quran verses", unit="verse"):
            ayah_text = verse['ayah_text']
            normalized_text = self.normalize_text(ayah_text)
            
            # Skip empty verses
            if not ayah_text or not normalized_text:
                continue
            
            verse_id = f"{verse['surah_id']}:{verse['ayah_id']}"
            verse_data = {
                'id': verse_id,
                'original_text': ayah_text,
                'normalized_text': normalized_text,
                'surah_id': verse['surah_id'],
                'surah_name': verse['surah_name'],
                'ayah_id': verse['ayah_id'],
                'type': 'ayah'
            }
            
            # Exact match index
            exact_hash = self.get_text_hash(ayah_text)
            self.quran_index['exact'][exact_hash] = verse_data
            
            # Normalized text index
            norm_hash = self.get_text_hash(normalized_text)
            self.quran_index['normalized'][norm_hash] = verse_data
            
            # Length-based buckets (for quick filtering)
            text_length = len(normalized_text)
            length_bucket = (text_length // 10) * 10  # Bucket by 10s
            self.quran_index['length_buckets'][length_bucket].append(verse_data)
            
            # N-gram index for fuzzy matching
            ngrams = self.generate_ngrams(normalized_text)
            for ngram in ngrams:
                self.quran_index['ngrams'][ngram].add(verse_id)
            
            # FuzzyWuzzy preparation
            if FUZZYWUZZY_AVAILABLE:
                self.quran_index['fuzzy_texts'].append((normalized_text, verse_data))
            
            # Whoosh full-text index
            if WHOOSH_AVAILABLE:
                quran_writer.add_document(
                    id=verse_id,
                    original_text=ayah_text,
                    normalized_text=normalized_text,
                    text_length=text_length,
                    type='ayah'
                )
        
        if WHOOSH_AVAILABLE:
            quran_writer.commit()
        
        # Build embeddings index
        if self.cohere_client and self.qdrant_client:
            print("Building Quran embeddings index...")
            all_data = list(self.quran_index['exact'].values())
            batch_size = 50
            total_batches = (len(all_data) + batch_size - 1) // batch_size
            
            with tqdm(total=total_batches, desc="Processing Quran embeddings", unit="batch") as pbar:
                for i in range(0, len(all_data), batch_size):
                    batch_data = all_data[i:i+batch_size]
                    texts = [data['normalized_text'] for data in batch_data]
                    
                    # Generate embeddings for this batch
                    embeddings = self.generate_embeddings_batch(texts, batch_size=len(texts))
                    
                    # Create points and insert immediately
                    if embeddings:
                        points = []
                        for j, (data, embedding) in enumerate(zip(batch_data, embeddings)):
                            if embedding is not None:
                                points.append(PointStruct(
                                    id=i + j,
                                    vector=embedding,
                                    payload={
                                        'text_id': data['id'],
                                        'original_text': data['original_text'],
                                        'normalized_text': data['normalized_text'],
                                        'surah_id': data['surah_id'],
                                        'surah_name': data['surah_name'],
                                        'ayah_id': data['ayah_id'],
                                        'verse_reference': f"Surah {data['surah_name']} ({data['surah_id']}:{data['ayah_id']})",
                                        'text_type': 'ayah',
                                        'text_length': len(data['normalized_text']),
                                        'word_count': len(data['normalized_text'].split())
                                    }
                                ))
                        
                        if points:
                            self.qdrant_client.upsert(
                                collection_name="quran_embeddings",
                                points=points
                            )
                    
                    pbar.update(1)
            
            print(f"✓ Indexed Quran embeddings in batches")
        
        print(f"✓ Indexed {len(quran_data)} Quran verses")
    
    def build_hadith_index(self, hadith_path="../datasets/raw-data/nine_books_data.csv"):
        """Build search index for Hadith texts"""
        print("Building Hadith index...")
        
        # Initialize Whoosh index
        if WHOOSH_AVAILABLE:
            schema = self.create_whoosh_schema()
            hadith_index_dir = Path("../datasets/indices/hadith_whoosh")
            hadith_index_dir.mkdir(parents=True, exist_ok=True)
            self.hadith_whoosh_index = create_in(str(hadith_index_dir), schema)
            hadith_writer = self.hadith_whoosh_index.writer()
        
        # Get total number of rows for progress bar
        print("Counting Hadith entries...")
        total_rows = sum(1 for _ in pd.read_csv(hadith_path, chunksize=10000))
        
        # Read CSV in chunks to handle large file
        chunk_size = 10000
        hadith_count = 0
        
        # Progress bar for chunks
        chunk_progress = tqdm(total=total_rows, desc="Processing Hadith chunks", unit="chunk")
        
        for chunk in pd.read_csv(hadith_path, chunksize=chunk_size, encoding='utf-8'):
            # Progress bar for rows within each chunk
            for _, row in tqdm(chunk.iterrows(), desc=f"Chunk {hadith_count//chunk_size + 1}", leave=False, total=len(chunk)):
                # Use Matn (actual hadith text) for indexing
                hadith_text = str(row.get('Matn', ''))
                
                # Skip empty or invalid hadiths
                if pd.isna(hadith_text) or not hadith_text.strip():
                    continue
                
                normalized_text = self.normalize_text(hadith_text)
                if not normalized_text:
                    continue
                
                hadith_id = str(row.get('hadithID', hadith_count))
                hadith_data = {
                    'id': hadith_id,
                    'original_text': hadith_text,
                    'normalized_text': normalized_text,
                    'book_id': str(row.get('BookID', '')),
                    'title': str(row.get('title', '')),
                    'type': 'hadith'
                }
                
                # Exact match index
                exact_hash = self.get_text_hash(hadith_text)
                self.hadith_index['exact'][exact_hash] = hadith_data
                
                # Normalized text index
                norm_hash = self.get_text_hash(normalized_text)
                self.hadith_index['normalized'][norm_hash] = hadith_data
                
                # Length-based buckets
                text_length = len(normalized_text)
                length_bucket = (text_length // 20) * 20  # Larger buckets for hadiths
                self.hadith_index['length_buckets'][length_bucket].append(hadith_data)
                
                # N-gram index
                ngrams = self.generate_ngrams(normalized_text)
                for ngram in ngrams:
                    self.hadith_index['ngrams'][ngram].add(hadith_id)
                
                # FuzzyWuzzy preparation
                if FUZZYWUZZY_AVAILABLE:
                    self.hadith_index['fuzzy_texts'].append((normalized_text, hadith_data))
                
                # Whoosh full-text index
                if WHOOSH_AVAILABLE:
                    hadith_writer.add_document(
                        id=hadith_id,
                        original_text=hadith_text,
                        normalized_text=normalized_text,
                        text_length=text_length,
                        type='hadith'
                    )
                
                hadith_count += 1
            
            chunk_progress.update(1)
        
        chunk_progress.close()
        
        if WHOOSH_AVAILABLE:
            print("Committing Whoosh index...")
            hadith_writer.commit()
        
        # Build embeddings index for Hadith
        if self.cohere_client and self.qdrant_client:
            print("Building Hadith embeddings index...")
            all_data = list(self.hadith_index['exact'].values())
            batch_size = 50
            total_batches = (len(all_data) + batch_size - 1) // batch_size
            
            with tqdm(total=total_batches, desc="Processing Hadith embeddings", unit="batch") as pbar:
                for i in range(0, len(all_data), batch_size):
                    batch_data = all_data[i:i+batch_size]
                    texts = [data['normalized_text'] for data in batch_data]
                    
                    # Generate embeddings for this batch
                    embeddings = self.generate_embeddings_batch(texts, batch_size=len(texts))
                    
                    # Create points and insert immediately
                    if embeddings:
                        points = []
                        for j, (data, embedding) in enumerate(zip(batch_data, embeddings)):
                            if embedding is not None:
                                points.append(PointStruct(
                                    id=i + j,
                                    vector=embedding,
                                    payload={
                                        'text_id': data['id'],
                                        'original_text': data['original_text'],
                                        'normalized_text': data['normalized_text'],
                                        'book_id': data['book_id'],
                                        'title': data.get('title', ''),
                                        'hadith_reference': f"Hadith {data['id']} (Book: {data['book_id']})",
                                        'text_type': 'hadith',
                                        'text_length': len(data['normalized_text']),
                                        'word_count': len(data['normalized_text'].split())
                                    }
                                ))
                        
                        if points:
                            self.qdrant_client.upsert(
                                collection_name="hadith_embeddings",
                                points=points
                            )
                    
                    pbar.update(1)
            
            print(f"✓ Indexed Hadith embeddings in batches")
        
        print(f"✓ Indexed {hadith_count} Hadith texts")
    
    def save_indices(self, output_dir="../datasets/indices"):
        """Save built indices to disk"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("Saving indices...")
        
        # Save Quran index
        quran_file = Path(output_dir) / "quran_search_index.pkl"
        with open(quran_file, 'wb') as f:
            pickle.dump(self.quran_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Quran index saved")
        
        # Save Hadith index
        hadith_file = Path(output_dir) / "hadith_search_index.pkl"
        with open(hadith_file, 'wb') as f:
            pickle.dump(self.hadith_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Hadith index saved")
        
        # Save metadata
        metadata = {
            'quran_entries': len(self.quran_index['exact']),
            'hadith_entries': len(self.hadith_index['exact']),
            'quran_ngrams': len(self.quran_index['ngrams']),
            'hadith_ngrams': len(self.hadith_index['ngrams']),
            'quran_fuzzy_texts': len(self.quran_index['fuzzy_texts']) if FUZZYWUZZY_AVAILABLE else 0,
            'hadith_fuzzy_texts': len(self.hadith_index['fuzzy_texts']) if FUZZYWUZZY_AVAILABLE else 0,
            'whoosh_enabled': WHOOSH_AVAILABLE,
            'fuzzywuzzy_enabled': FUZZYWUZZY_AVAILABLE,
            'cohere_enabled': COHERE_AVAILABLE and self.cohere_client is not None,
            'qdrant_enabled': QDRANT_AVAILABLE and self.qdrant_client is not None,
            'embedding_dimension': self.embedding_dimension,
        }
        
        metadata_file = Path(output_dir) / "index_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print("✓ Metadata saved")
        
        print(f"\n🎉 All indices saved to {output_dir}")
        print(f"📊 Quran entries: {metadata['quran_entries']:,}")
        print(f"📊 Hadith entries: {metadata['hadith_entries']:,}")
        if WHOOSH_AVAILABLE:
            print("🔍 Whoosh full-text search: Enabled")
        if FUZZYWUZZY_AVAILABLE:
            print("🎯 FuzzyWuzzy matching: Enabled")

def main():
    """Build and save religious text search indices"""
    parser = argparse.ArgumentParser(description='Build religious text search indices')
    parser.add_argument('--do-qdrant-reset', action='store_true',
                        help='Reset (delete and recreate) Qdrant collections before indexing')
    
    args = parser.parse_args()
    
    print("🚀 Starting Religious Text Index Builder")
    print("=" * 50)
    
    if args.do_qdrant_reset:
        print("⚠️  Qdrant reset mode enabled - collections will be deleted and recreated")
    
    indexer = ReligiousTextIndexer()
    
    # Setup Qdrant collections if available
    if indexer.qdrant_client:
        print("Setting up Qdrant collections...")
        indexer.setup_qdrant_collections(reset=args.do_qdrant_reset)
    
    # Build indices
    indexer.build_quran_index()
    indexer.build_hadith_index()
    
    # Save to disk
    indexer.save_indices()
    
    print("\n" + "=" * 50)
    print("🏆 Index building completed successfully!")
    print("Ready for gold prize competition! 🥇")

if __name__ == "__main__":
    main()
