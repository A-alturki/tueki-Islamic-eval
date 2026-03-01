#!/usr/bin/env python3
"""
Religious Text Search Engine
Fast hierarchical search for Quran verses and Hadith texts with multiple matching strategies.
"""

import json
import pickle
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

try:
    import pyarabic.araby as araby
except ImportError:
    def normalize_arabic(text):
        return re.sub(r'[^\u0600-\u06FF\s]', '', text).strip()
    araby = type('obj', (object,), {'strip_diacritics': normalize_arabic, 'normalize_hamza': lambda x: x})()

try:
    from whoosh.index import open_dir
    from whoosh.qparser import QueryParser
    from whoosh import scoring
    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import textdistance
    TEXTDISTANCE_AVAILABLE = True
except ImportError:
    TEXTDISTANCE_AVAILABLE = False

try:
    import cohere
    from dotenv import dotenv_values
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

@dataclass
class SearchResult:
    """Container for search results"""
    text_id: str
    original_text: str
    normalized_text: str
    match_type: str  # 'exact', 'normalized', 'whoosh', 'fuzzy', 'ngram'
    confidence: float
    text_type: str  # 'ayah' or 'hadith'
    metadata: Dict
    
    # Ayah-specific properties
    @property
    def surah_id(self) -> Optional[int]:
        """Get surah ID for ayah results"""
        return self.metadata.get('surah_id') if self.text_type == 'ayah' else None
    
    @property
    def surah_name(self) -> Optional[str]:
        """Get surah name for ayah results"""
        return self.metadata.get('surah_name') if self.text_type == 'ayah' else None
    
    @property
    def ayah_id(self) -> Optional[int]:
        """Get ayah ID for ayah results"""
        return self.metadata.get('ayah_id') if self.text_type == 'ayah' else None
    
    @property
    def book_id(self) -> Optional[str]:
        """Get book ID for hadith results"""
        return self.metadata.get('book_id') if self.text_type == 'hadith' else None
    
    @property
    def hadith_title(self) -> Optional[str]:
        """Get title for hadith results"""
        return self.metadata.get('title') if self.text_type == 'hadith' else None

class ReligiousTextSearcher:
    """
    Hierarchical search engine for religious texts with multiple matching strategies
    """
    
    def __init__(self, indices_dir="../datasets/indices"):
        self.indices_dir = Path(indices_dir)
        self.quran_index = None
        self.hadith_index = None
        self.quran_whoosh = None
        self.hadith_whoosh = None
        self.metadata = None
        
        # Embedding search components
        self.cohere_client = None
        self.qdrant_client = None
        
        # Search thresholds
        self.fuzzy_threshold = 85
        self.ngram_threshold = 0.7
        self.whoosh_limit = 15  # Increased for re-ranking
        self.embedding_limit = 35  # Increased for re-ranking
        
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
                self.qdrant_client = QdrantClient(path=str(self.indices_dir / "qdrant_db"))
            except Exception as e:
                print(f"Warning: Could not initialize Qdrant client: {e}")
        
    def load_indices(self):
        """Load all search indices from disk"""
        try:
            # Load pickle indices
            with open(self.indices_dir / "quran_search_index.pkl", 'rb') as f:
                self.quran_index = pickle.load(f)
            
            with open(self.indices_dir / "hadith_search_index.pkl", 'rb') as f:
                self.hadith_index = pickle.load(f)
            
            # Load metadata
            with open(self.indices_dir / "index_metadata.json", 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load Whoosh indices if available
            if WHOOSH_AVAILABLE and self.metadata.get('whoosh_enabled', False):
                try:
                    self.quran_whoosh = open_dir(str(self.indices_dir / "quran_whoosh"))
                    self.hadith_whoosh = open_dir(str(self.indices_dir / "hadith_whoosh"))
                except:
                    print("Warning: Could not load Whoosh indices")
            
            print(f"✓ Loaded indices: {self.metadata['quran_entries']:,} Quran + {self.metadata['hadith_entries']:,} Hadith")
            return True
            
        except Exception as e:
            print(f"Error loading indices: {e}")
            return False
    
    def normalize_text(self, text: str) -> str:
        """Normalize Arabic text for matching"""
        if not text:
            return ""
        
        normalized = araby.strip_diacritics(text)
        normalized = araby.normalize_hamza(normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\u0600-\u06FF\s]', '', normalized)
        
        return normalized.strip()
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for exact matching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def filter_by_book_ids(self, results: List[SearchResult], book_ids: List[str]) -> List[SearchResult]:
        """Filter hadith results by book IDs"""
        if not book_ids:
            return results
        
        filtered_results = []
        for result in results:
            if result.text_type == 'ayah':
                # Always include Ayah results (no book filtering)
                filtered_results.append(result)
            elif result.book_id and result.book_id in book_ids:
                # Include hadith results only if book_id matches
                filtered_results.append(result)
        
        return filtered_results
    
    def search_exact(self, query: str, search_type: str = 'both') -> List[SearchResult]:
        """Level 1: Exact hash-based search (fastest)"""
        results = []
        query_hash = self.get_text_hash(query)
        
        if search_type in ['both', 'ayah'] and self.quran_index:
            if query_hash in self.quran_index['exact']:
                data = self.quran_index['exact'][query_hash]
                results.append(SearchResult(
                    text_id=data['id'],
                    original_text=data['original_text'],
                    normalized_text=data['normalized_text'],
                    match_type='exact',
                    confidence=1.0,
                    text_type='ayah',
                    metadata=data
                ))
        
        if search_type in ['both', 'hadith'] and self.hadith_index:
            if query_hash in self.hadith_index['exact']:
                data = self.hadith_index['exact'][query_hash]
                results.append(SearchResult(
                    text_id=data['id'],
                    original_text=data['original_text'],
                    normalized_text=data['normalized_text'],
                    match_type='exact',
                    confidence=1.0,
                    text_type='hadith',
                    metadata=data
                ))
        
        return results
    
    def search_normalized(self, query: str, search_type: str = 'both') -> List[SearchResult]:
        """Level 2: Normalized text search (fast)"""
        results = []
        normalized_query = self.normalize_text(query)
        query_hash = self.get_text_hash(normalized_query)
        
        if search_type in ['both', 'ayah'] and self.quran_index:
            if query_hash in self.quran_index['normalized']:
                data = self.quran_index['normalized'][query_hash]
                results.append(SearchResult(
                    text_id=data['id'],
                    original_text=data['original_text'],
                    normalized_text=data['normalized_text'],
                    match_type='normalized',
                    confidence=0.95,
                    text_type='ayah',
                    metadata=data
                ))
        
        if search_type in ['both', 'hadith'] and self.hadith_index:
            if query_hash in self.hadith_index['normalized']:
                data = self.hadith_index['normalized'][query_hash]
                results.append(SearchResult(
                    text_id=data['id'],
                    original_text=data['original_text'],
                    normalized_text=data['normalized_text'],
                    match_type='normalized',
                    confidence=0.95,
                    text_type='hadith',
                    metadata=data
                ))
        
        return results
    
    def search_whoosh(self, query: str, search_type: str = 'both') -> List[SearchResult]:
        """Level 3: Whoosh full-text search (medium speed)"""
        if not WHOOSH_AVAILABLE or not self.quran_whoosh or not self.hadith_whoosh:
            return []
        
        results = []
        normalized_query = self.normalize_text(query)
        
        try:
            if search_type in ['both', 'ayah']:
                with self.quran_whoosh.searcher() as searcher:
                    parser = QueryParser("normalized_text", self.quran_whoosh.schema)
                    whoosh_query = parser.parse(normalized_query)
                    hits = searcher.search(whoosh_query, limit=self.whoosh_limit)
                    
                    for hit in hits:
                        # Get full metadata from original index using text_id
                        text_id = hit['id']
                        full_data = None
                        
                        # Search in original index for complete metadata
                        for data in self.quran_index['exact'].values():
                            if data['id'] == text_id:
                                full_data = data
                                break
                        
                        if not full_data:
                            # Fallback: search in normalized index
                            for data in self.quran_index['normalized'].values():
                                if data['id'] == text_id:
                                    full_data = data
                                    break
                        
                        if full_data:
                            results.append(SearchResult(
                                text_id=text_id,
                                original_text=full_data['original_text'],
                                normalized_text=full_data['normalized_text'],
                                match_type='whoosh',
                                confidence=float(hit.score),
                                text_type='ayah',
                                metadata=full_data
                            ))
            
            if search_type in ['both', 'hadith']:
                with self.hadith_whoosh.searcher() as searcher:
                    parser = QueryParser("normalized_text", self.hadith_whoosh.schema)
                    whoosh_query = parser.parse(normalized_query)
                    hits = searcher.search(whoosh_query, limit=self.whoosh_limit)
                    
                    for hit in hits:
                        # Get full metadata from original index using text_id
                        text_id = hit['id']
                        full_data = None
                        
                        # Search in original index for complete metadata
                        for data in self.hadith_index['exact'].values():
                            if data['id'] == text_id:
                                full_data = data
                                break
                        
                        if not full_data:
                            # Fallback: search in normalized index
                            for data in self.hadith_index['normalized'].values():
                                if data['id'] == text_id:
                                    full_data = data
                                    break
                        
                        if full_data:
                            results.append(SearchResult(
                                text_id=text_id,
                                original_text=full_data['original_text'],
                                normalized_text=full_data['normalized_text'],
                                match_type='whoosh',
                                confidence=float(hit.score),
                                text_type='hadith',
                                metadata=full_data
                            ))
        except Exception as e:
            print(f"Whoosh search error: {e}")
        
        return results
    
    def search_fuzzy(self, query: str, search_type: str = 'both', limit: int = 5) -> List[SearchResult]:
        """Level 4: FuzzyWuzzy search (slower but accurate)"""
        if not FUZZYWUZZY_AVAILABLE:
            return []
        
        results = []
        normalized_query = self.normalize_text(query)
        
        # For partial matching, also try with partial_ratio which is better for substring matching
        if search_type in ['both', 'ayah'] and self.quran_index.get('fuzzy_texts'):
            choices = [text for text, _ in self.quran_index['fuzzy_texts']]
            
            # Try both regular ratio and partial ratio
            matches_regular = process.extract(normalized_query, choices, limit=limit, scorer=fuzz.ratio)
            matches_partial = process.extract(normalized_query, choices, limit=limit, scorer=fuzz.partial_ratio)
            matches_token = process.extract(normalized_query, choices, limit=limit, scorer=fuzz.token_sort_ratio)
            
            # Combine and deduplicate results
            all_matches = {}
            for match_text, score in matches_regular + matches_partial + matches_token:
                if match_text not in all_matches or all_matches[match_text] < score:
                    all_matches[match_text] = score
            
            # Sort by score and take top results
            sorted_matches = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)
            
            for match_text, score in sorted_matches[:limit]:
                if score >= max(70, self.fuzzy_threshold - 15):  # Lower threshold for partial matches
                    # Find the corresponding data
                    for text, data in self.quran_index['fuzzy_texts']:
                        if text == match_text:
                            results.append(SearchResult(
                                text_id=data['id'],
                                original_text=data['original_text'],
                                normalized_text=data['normalized_text'],
                                match_type='fuzzy',
                                confidence=score / 100.0,
                                text_type='ayah',
                                metadata=data
                            ))
                            break
        
        if search_type in ['both', 'hadith'] and self.hadith_index.get('fuzzy_texts'):
            choices = [text for text, _ in self.hadith_index['fuzzy_texts']]
            
            # Try both regular ratio and partial ratio
            matches_regular = process.extract(normalized_query, choices, limit=limit, scorer=fuzz.ratio)
            matches_partial = process.extract(normalized_query, choices, limit=limit, scorer=fuzz.partial_ratio)
            matches_token = process.extract(normalized_query, choices, limit=limit, scorer=fuzz.token_sort_ratio)
            
            # Combine and deduplicate results
            all_matches = {}
            for match_text, score in matches_regular + matches_partial + matches_token:
                if match_text not in all_matches or all_matches[match_text] < score:
                    all_matches[match_text] = score
            
            # Sort by score and take top results
            sorted_matches = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)
            
            for match_text, score in sorted_matches[:limit]:
                if score >= max(70, self.fuzzy_threshold - 15):  # Lower threshold for partial matches
                    # Find the corresponding data
                    for text, data in self.hadith_index['fuzzy_texts']:
                        if text == match_text:
                            results.append(SearchResult(
                                text_id=data['id'],
                                original_text=data['original_text'],
                                normalized_text=data['normalized_text'],
                                match_type='fuzzy',
                                confidence=score / 100.0,
                                text_type='hadith',
                                metadata=data
                            ))
                            break
        
        return results
    
    def search_partial(self, query: str, search_type: str = 'both', limit: int = 10) -> List[SearchResult]:
        """Level 4.5: Partial matching for substring queries with word substitutions"""
        if not FUZZYWUZZY_AVAILABLE:
            return []
        
        results = []
        normalized_query = self.normalize_text(query)
        query_words = normalized_query.split()
        
        # Only proceed if query has meaningful words (more than 2 words)
        if len(query_words) < 3:
            return results
        
        candidates = []
        
        if search_type in ['both', 'ayah'] and self.quran_index.get('fuzzy_texts'):
            candidates.extend([(text, data, 'ayah') for text, data in self.quran_index['fuzzy_texts']])
        
        if search_type in ['both', 'hadith'] and self.hadith_index.get('fuzzy_texts'):
            candidates.extend([(text, data, 'hadith') for text, data in self.hadith_index['fuzzy_texts']])
        
        # Score candidates based on word overlap and partial ratio
        scored_candidates = []
        
        for text, data, text_type in candidates:
            text_words = text.split()
            
            # Calculate word overlap ratio
            common_words = set(query_words) & set(text_words)
            word_overlap = len(common_words) / len(query_words) if query_words else 0
            
            # Calculate partial ratio for substring matching
            partial_score = fuzz.partial_ratio(normalized_query, text)
            
            # Calculate token sort ratio for word order flexibility
            token_score = fuzz.token_sort_ratio(normalized_query, text)
            
            # Combine scores with weights
            combined_score = (word_overlap * 30) + (partial_score * 0.4) + (token_score * 0.3)
            
            if combined_score >= 50:  # Threshold for partial matches
                scored_candidates.append((combined_score, text, data, text_type))
        
        # Sort by score and take top results
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        for score, text, data, text_type in scored_candidates[:limit]:
            results.append(SearchResult(
                text_id=data['id'],
                original_text=data['original_text'],
                normalized_text=data['normalized_text'],
                match_type='partial',
                confidence=min(score / 100.0, 0.95),  # Cap at 0.95 since it's partial
                text_type=text_type,
                metadata=data
            ))
        
        return results
    
    def search_ngram(self, query: str, search_type: str = 'both', limit: int = 10) -> List[SearchResult]:
        """Level 5: N-gram similarity search (fallback)"""
        if not TEXTDISTANCE_AVAILABLE:
            return []
        
        results = []
        normalized_query = self.normalize_text(query)
        query_length = len(normalized_query)
        
        # Length-based filtering for efficiency
        length_range = range(max(1, query_length - 20), query_length + 21)
        
        candidates = []
        
        if search_type in ['both', 'ayah'] and self.quran_index:
            for length in length_range:
                bucket_key = (length // 10) * 10
                if bucket_key in self.quran_index['length_buckets']:
                    candidates.extend(self.quran_index['length_buckets'][bucket_key])
        
        if search_type in ['both', 'hadith'] and self.hadith_index:
            for length in length_range:
                bucket_key = (length // 20) * 20
                if bucket_key in self.hadith_index['length_buckets']:
                    candidates.extend(self.hadith_index['length_buckets'][bucket_key])
        
        # Calculate similarities
        similarities = []
        for candidate in candidates[:1000]:  # Limit for performance
            similarity = textdistance.jaccard(normalized_query, candidate['normalized_text'])
            if similarity >= self.ngram_threshold:
                similarities.append((similarity, candidate))
        
        # Sort by similarity and take top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        for similarity, data in similarities[:limit]:
            results.append(SearchResult(
                text_id=data['id'],
                original_text=data['original_text'],
                normalized_text=data['normalized_text'],
                match_type='ngram',
                confidence=similarity,
                text_type=data['type'],
                metadata={'jaccard_score': similarity}
            ))
        
        return results
    
    def search_embeddings(self, query: str, search_type: str = 'both', limit: int = 10) -> List[SearchResult]:
        """Embedding-based semantic search using Cohere + Qdrant"""
        if not self.cohere_client or not self.qdrant_client:
            return []
        
        results = []
        normalized_query = self.normalize_text(query)
        
        try:
            # Generate query embedding
            text_input = {"content": [{"type": "text", "text": normalized_query}]}
            response = self.cohere_client.embed(
                inputs=[text_input],
                model="embed-v4.0",
                input_type="search_query",
                embedding_types=["float"],
            )
            query_embedding = response.embeddings.float[0]
            
            # Search in Quran collection
            if search_type in ['both', 'ayah']:
                try:
                    hits = self.qdrant_client.query_points(
                        collection_name="quran_embeddings",
                        query=query_embedding,
                        limit=limit
                    )
                    
                    for hit in hits.points:
                        payload = hit.payload
                        results.append(SearchResult(
                            text_id=payload['text_id'],
                            original_text=payload['original_text'],
                            normalized_text=payload['normalized_text'],
                            match_type='embedding',
                            confidence=float(hit.score),
                            text_type='ayah',
                            metadata={
                                'surah_id': payload['surah_id'],
                                'surah_name': payload['surah_name'],
                                'ayah_id': payload['ayah_id'],
                                'verse_reference': payload.get('verse_reference', ''),
                                'text_length': payload.get('text_length', 0),
                                'word_count': payload.get('word_count', 0)
                            }
                        ))
                except Exception as e:
                    print(f"Error searching Quran embeddings: {e}")
            
            # Search in Hadith collection
            if search_type in ['both', 'hadith']:
                try:
                    hits = self.qdrant_client.query_points(
                        collection_name="hadith_embeddings",
                        query=query_embedding,
                        limit=limit
                    )
                    
                    for hit in hits.points:
                        payload = hit.payload
                        results.append(SearchResult(
                            text_id=payload['text_id'],
                            original_text=payload['original_text'],
                            normalized_text=payload['normalized_text'],
                            match_type='embedding',
                            confidence=float(hit.score),
                            text_type='hadith',
                            metadata={
                                'book_id': payload['book_id'],
                                'title': payload.get('title', ''),
                                'hadith_reference': payload.get('hadith_reference', ''),
                                'text_length': payload.get('text_length', 0),
                                'word_count': payload.get('word_count', 0)
                            }
                        ))
                except Exception as e:
                    print(f"Error searching Hadith embeddings: {e}")
                    
        except Exception as e:
            print(f"Error in embedding search: {e}")
        
        return results
    
    def rerank_results(self, query: str, results: List[SearchResult], top_n: int = 15, min_score: float = 0.15) -> List[SearchResult]:
        """Re-rank search results using Cohere rerank model"""
        if not self.cohere_client or len(results) <= 1:
            return results[:top_n]
        
        try:
            # Extract documents for re-ranking
            docs = [result.original_text for result in results]
            
            # Call Cohere rerank API
            response = self.cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs,
                top_n=min(len(docs), 50)  # Rerank up to 50 docs
            )
            
            # Filter by relevance score and create reranked results
            reranked_results = []
            for result_item in response.results:
                if result_item.relevance_score > min_score:
                    original_result = results[result_item.index]
                    # Update confidence with rerank score
                    original_result.confidence = result_item.relevance_score
                    original_result.match_type = f"{original_result.match_type}_reranked"
                    reranked_results.append(original_result)
            
            # Return reranked results if we have good matches, otherwise fallback
            if reranked_results:
                return reranked_results[:top_n]
            else:
                # Fallback: return top 20 from original results
                return results[:20]
                
        except Exception as e:
            print(f"Rerank error: {e}")
            return results[:top_n]
    
    def search_hierarchical(self, query: str, search_type: str = 'both', max_results: int = 10, book_ids: List[str] = None) -> Tuple[List[SearchResult], int]:
        """
        Enhanced hierarchical search with Cohere re-ranking for optimal results
        Returns: (results, total_found_count)
        
        Args:
            query: Search query text
            search_type: 'both', 'ayah', or 'hadith'  
            max_results: Maximum number of results to return
            book_ids: List of book IDs to filter hadith results (default: ["1.0", "2.0"])
        """
        if book_ids is None:
            book_ids = ["1.0", "2.0"]  # Default to first two hadith books
        
        all_results = []
        seen_ids = set()
        
        # Level 1: Exact match (highest priority, no re-ranking needed)
        exact_results = self.search_exact(query, search_type)
        exact_results = self.filter_by_book_ids(exact_results, book_ids)
        for result in exact_results:
            if result.text_id not in seen_ids:
                all_results.append(result)
                seen_ids.add(result.text_id)
        
        # Level 2: Normalized match
        if len(all_results) < max_results:
            normalized_results = self.search_normalized(query, search_type)
            normalized_results = self.filter_by_book_ids(normalized_results, book_ids)
            for result in normalized_results:
                if result.text_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.text_id)
        
        # Return early if we have exact/normalized matches
        if all_results:
            return all_results[:max_results], len(all_results)
        
        # Level 3: Combined search with re-ranking for fuzzy matches
        combined_results = []
        
        # Get more results for re-ranking: 35 from embeddings + 15 from whoosh
        if self.cohere_client:
            embedding_results = self.search_embeddings(query, search_type, limit=35)
            embedding_results = self.filter_by_book_ids(embedding_results, book_ids)
            whoosh_results = self.search_whoosh(query, search_type)
            whoosh_results = self.filter_by_book_ids(whoosh_results, book_ids)
            
            # Combine and deduplicate
            for result in embedding_results + whoosh_results:
                if result.text_id not in seen_ids:
                    combined_results.append(result)
                    seen_ids.add(result.text_id)
            
            # Apply re-ranking if we have enough results
            if combined_results:
                reranked_results = self.rerank_results(query, combined_results, top_n=max_results)
                return reranked_results, len(combined_results)
        
        # Fallback: Original hierarchical approach
        # Level 3: Whoosh + Embedding without re-ranking
        whoosh_results = self.search_whoosh(query, search_type) if len(all_results) < max_results else []
        whoosh_results = self.filter_by_book_ids(whoosh_results, book_ids)
        embedding_results = self.search_embeddings(query, search_type, limit=max_results) if len(all_results) < max_results else []
        embedding_results = self.filter_by_book_ids(embedding_results, book_ids)
        
        # Combine and sort by confidence
        combined_results = whoosh_results + embedding_results
        combined_results.sort(key=lambda x: x.confidence, reverse=True)
        
        for result in combined_results:
            if result.text_id not in seen_ids and len(all_results) < max_results:
                all_results.append(result)
                seen_ids.add(result.text_id)
        
        # Return early if we have good results
        if all_results:
            return all_results[:max_results], len(all_results)
        
        # Level 4: Fuzzy matching with multiple scorers
        fuzzy_results = self.search_fuzzy(query, search_type, limit=max_results*2)
        fuzzy_results = self.filter_by_book_ids(fuzzy_results, book_ids)
        for result in fuzzy_results:
            if result.text_id not in seen_ids and len(all_results) < max_results:
                all_results.append(result)
                seen_ids.add(result.text_id)
        
        if all_results:
            return all_results[:max_results], len(all_results)
        
        # Level 5: Partial matching for substring queries
        partial_results = self.search_partial(query, search_type, limit=max_results*2)
        partial_results = self.filter_by_book_ids(partial_results, book_ids)
        for result in partial_results:
            if result.text_id not in seen_ids and len(all_results) < max_results:
                all_results.append(result)
                seen_ids.add(result.text_id)
        
        if all_results:
            return all_results[:max_results], len(all_results)
        
        # Level 6: N-gram similarity (last resort)
        ngram_results = self.search_ngram(query, search_type, limit=max_results*2)
        ngram_results = self.filter_by_book_ids(ngram_results, book_ids)
        for result in ngram_results:
            if result.text_id not in seen_ids and len(all_results) < max_results:
                all_results.append(result)
                seen_ids.add(result.text_id)
        
        return all_results[:max_results], len(all_results)
    
    def classify_text(self, query: str) -> str:
        """
        Classify text as CorrectAyah, WrongAyah, CorrectHadith, or WrongHadith
        """
        # Search both types and get multiple results for better analysis
        results, _ = self.search_hierarchical(query, 'both', max_results=5)
        
        if not results:
            # No match found - likely wrong
            # Try to determine if it's claiming to be ayah or hadith based on patterns
            if self._looks_like_ayah(query):
                return "WrongAyah"
            else:
                return "WrongHadith"
        
        # Smart classification based on content similarity, not just confidence
        normalized_query = self.normalize_text(query)
        query_length = len(normalized_query)
        
        best_ayah_match = None
        best_hadith_match = None
        
        for result in results:
            # Calculate semantic similarity (length and word overlap)
            result_length = len(result.normalized_text)
            length_ratio = min(query_length, result_length) / max(query_length, result_length)
            
            # Skip very short matches that are likely false positives
            if result_length < 10 and length_ratio < 0.3:
                continue
                
            # Calculate word overlap for better semantic matching
            query_words = set(normalized_query.split())
            result_words = set(result.normalized_text.split())
            word_overlap = len(query_words & result_words) / len(query_words) if query_words else 0
            
            # Combined semantic score (confidence + length similarity + word overlap)
            semantic_score = (result.confidence * 0.4) + (length_ratio * 0.3) + (word_overlap * 0.3)
            
            if result.text_type == 'ayah':
                if not best_ayah_match or semantic_score > best_ayah_match[1]:
                    best_ayah_match = (result, semantic_score)
            else:
                if not best_hadith_match or semantic_score > best_hadith_match[1]:
                    best_hadith_match = (result, semantic_score)
        
        # Choose the best match based on semantic score
        if best_ayah_match and best_hadith_match:
            if best_ayah_match[1] > best_hadith_match[1]:
                result, score = best_ayah_match
            else:
                result, score = best_hadith_match
        elif best_ayah_match:
            result, score = best_ayah_match
        elif best_hadith_match:
            result, score = best_hadith_match
        else:
            # Fallback to first result
            result = results[0]
            score = result.confidence
        
        # Classification thresholds based on semantic similarity
        if score >= 0.7:  # High semantic similarity
            if result.text_type == 'ayah':
                return "CorrectAyah"
            else:
                return "CorrectHadith"
        else:
            # Lower similarity - likely wrong but still indicate type
            if result.text_type == 'ayah':
                return "WrongAyah"
            else:
                return "WrongHadith"
    
    def _looks_like_ayah(self, text: str) -> bool:
        """Heuristic to determine if text looks like an ayah"""
        # Simple heuristics - can be improved
        ayah_patterns = [
            r'قُل', r'يَا أَيُّهَا', r'وَ', r'إِنَّ', r'قَالَ',
            r'اللَّهُ', r'رَبِّ', r'وَمَا', r'فَ'
        ]
        
        normalized = self.normalize_text(text)
        ayah_score = sum(1 for pattern in ayah_patterns if re.search(pattern, normalized))
        
        return ayah_score >= 2  # Threshold can be tuned

# Example usage and testing
if __name__ == "__main__":
    searcher = ReligiousTextSearcher()
    
    if searcher.load_indices():
        print("🔍 Testing Religious Text Search Engine")
        print("=" * 60)
        
        # Test cases
        test_queries = [
            # "بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ",  # Exact Quran verse
            "الحمد للرب رب العالمين",                    # Partial Quran verse
            # "إنما الأعمال بالنيات",                    # Hadith text
            # "قل هو الله احد الله الصمد",                # Modified Quran verse
            # " الْمَرْأَةُ يَنْقَاطِيعُ لديها الدَّمُ أَيَأْتِيهَا",
            "لا تُناجَ أَحَدُكُمُ أَخَاهُ فيما يُحِبُّ، وَلَكِنَّهُ يُحِبُّهُ فيما يُحِبُّهُ",
            "الاعمال الصالحات دائما نحسبها بالنية الصالحة",
            "إِنَّمَا يَعْلَمُ سِرَّ الْأَصْفَى وَمَا تَدْرِي أَنَّهُمْ يَسْمَعُونَ}"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🧪 Test {i}: {query}")
            print("-" * 40)
            
            # Hierarchical search with result count
            results, total_found = searcher.search_hierarchical(query, max_results=7, search_type="hadith")
            
            print(f"📊 Found {len(results)} results (showing top 3 of {total_found} total)")
            
            if results:
                for j, result in enumerate(results, 1):
                    print(f"\n  {j}. Match Type: {result.match_type} | Confidence: {result.confidence:.3f}")
                    print(f"     Type: {result.text_type}")
                    
                    if result.text_type == 'ayah':
                        print(f"     📖 Surah: {result.surah_name} ({result.surah_id}), Ayah: {result.ayah_id}")
                    else:
                        print(f"     📚 Book ID: {result.book_id}")
                        if result.hadith_title:
                            print(f"     📚 Title: {result.hadith_title[:50]}...")
                    
                    print(f"     📝 Text: {result.original_text[:80]}...")
                
                # Classification test
                classification = searcher.classify_text(query)
                print(f"\n  🏷️  Classification: {classification}")
            else:
                print("  ❌ No results found")
        
        print("\n" + "=" * 60)
        print("🎯 Search engine ready for competition pipeline!")
        
        # Performance summary
        print(f"\n📈 Index Summary:")
        print(f"   • Quran entries: {searcher.metadata['quran_entries']:,}")
        print(f"   • Hadith entries: {searcher.metadata['hadith_entries']:,}")
        print(f"   • Whoosh enabled: {searcher.metadata.get('whoosh_enabled', False)}")
        print(f"   • FuzzyWuzzy enabled: {searcher.metadata.get('fuzzywuzzy_enabled', False)}")
        print(f"   • Cohere embeddings enabled: {searcher.metadata.get('cohere_enabled', False)}")
        print(f"   • Qdrant vector search enabled: {searcher.metadata.get('qdrant_enabled', False)}")
        
    else:
        print("❌ Failed to load indices. Please run the indexer first:")
        print("   python 07-index-religion-dataset-for-search.py")
