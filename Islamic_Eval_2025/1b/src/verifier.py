"""
Main verification logic for processing Islamic text queries.
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
import joblib
from tqdm import tqdm

from .config import Config
from .verse_merger import merge_ayas_from_retrieval
from .prompts import get_verification_prompt
from .text_processor import clean_text


class IslamicTextVerifier:
    """Main class for verifying Islamic text matches using LLM."""

    def __init__(self):
        """Initialize the verifier with configuration and LLM."""
        Config.validate_config()
        self.llm = Config.get_llm()
        self.prompt = get_verification_prompt()
        self.existing_results = {}

    def load_existing_results(self) -> None:
        """Load existing results from output file if it exists."""
        output_path = Config.get_output_file_path()
        if os.path.exists(output_path):
            try:
                existing_data = joblib.load(output_path)
                for item in existing_data:
                    composite_key = f"{item['sequence_id']}"
                    self.existing_results[composite_key] = item
                print(f"Loaded {len(self.existing_results)} existing results from {output_path}")
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")

    def load_input_data(self) -> List[Dict[str, Any]]:
        """Load input data from pickle file."""
        try:
            import pickle
            with open(Config.INPUT_FILE, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} queries from {Config.INPUT_FILE}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {Config.INPUT_FILE}")
        except Exception as e:
            raise ValueError(f"Error loading pickle file: {e}")

    def get_text_key(self, span_type: str) -> str:
        """Get the appropriate text key based on span type."""
        if span_type in ["WrongAyah", "CorrectAyah"]:
            return "ayah_text"
        elif span_type in ["WrongHadith", "CorrectHadith"]:
            return "hadithTxt"
        else:
            raise ValueError(f"Unknown span_type: {span_type}")

    def verify_text_match(self, query_text: str, candidate_text: str) -> bool:
        """
        Verify if candidate text matches query text using LLM.
        
        Args:
            query_text: The query text to match
            candidate_text: The candidate text to verify
            
        Returns:
            True if texts match, False otherwise
        """
        messages = self.prompt.format_prompt(
            query=query_text,
            text=candidate_text
        ).to_messages()

        response = self.llm.invoke(messages)
        answer = response.content.strip().lower()
        return answer == "true"

    def process_matches(self, query_text: str, match_list: List[Dict[str, Any]],
                        text_key: str, composite_key: str) -> List[Dict[str, Any]]:
        """
        Process a list of matches for a given query.
        
        Args:
            query_text: The original query text
            match_list: List of candidate matches
            text_key: Key to extract text from matches
            composite_key: Unique identifier for caching
            
        Returns:
            List of processed matches with detection results
        """
        processed_matches = []

        for match in match_list:
            candidate_text = match[text_key]

            # Check if we already have results for this text
            found_in_existing = []
            if composite_key in self.existing_results:
                found_in_existing = [
                    ent for ent in self.existing_results[composite_key]['matches']
                    if ent.get(text_key) == candidate_text and "detection" in ent
                ]

            if found_in_existing:
                # Reuse existing result
                match_with_detection = found_in_existing[0]
                processed_matches.append(match_with_detection)
                print("Reusing previous result")
            else:
                # Perform new verification
                query_text = clean_text(query_text, )
                candidate_text = clean_text(candidate_text, )
                detection = self.verify_text_match(query_text, candidate_text)

                match_with_detection = match.copy()
                match_with_detection["detection"] = detection
                processed_matches.append(match_with_detection)

                if detection:
                    print(f"✅ Match found!")
                    print(f"Query: {query_text}")
                    print(f"Candidate: {candidate_text}")
                    print("=" * 50)
                    break  # Stop at first match

        return processed_matches

    def process_single_query(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single query item.
        
        Args:
            item: Query item dictionary
            
        Returns:
            Processed result dictionary
        """
        sequence_id = item["sequence_id"]
        query_id = item["question_id"]
        query_text = item["query_text"]
        span_type = item["span_type"]
        match_list = item["top_20_match_details"]

        # Merge verses if dealing with Ayah
        if span_type == "Ayah":
            match_list = merge_ayas_from_retrieval(match_list)

        composite_key = f"{sequence_id}"
        is_merged = len(item["top_20_match_details"]) != len(match_list)

        # Check if already processed (skip only if not merged)
        if composite_key in self.existing_results and not is_merged:
            print(f"Skipping sequence_id {sequence_id} - already processed")
            return self.existing_results[composite_key]

        # Get appropriate text key
        text_key = self.get_text_key(span_type)

        # Process matches
        processed_matches = self.process_matches(
            query_text, match_list, text_key, composite_key
        )

        return {
            "id": query_id,
            "query": query_text,
            "sequence_id": sequence_id,
            "span_type": span_type,
            "matches": processed_matches,
        }

    def save_results(self, output: List[Dict[str, Any]],
                     filename: Optional[str] = None) -> None:
        """
        Save results to file in the appropriate results subfolder.
        
        Args:
            output: List of processed results
            filename: Optional custom filename
        """
        if filename is None:
            filepath = Config.get_output_file_path()
        else:
            filepath = Config.get_output_file_path(filename)

        joblib.dump(output, filepath, compress=5)
        print(f"💾 Results saved to {filepath}")

    def run(self) -> None:
        """Main execution method."""
        print("🚀 Starting Islamic Text Verification System")

        # Load data
        self.load_existing_results()
        data = self.load_input_data()

        output = []

        # Process each query
        for item in tqdm(data, desc="Processing queries"):
            try:
                result = self.process_single_query(item)
                output.append(result)

                # Save intermediate results with timestamp
                timestamp = int(time.time())
                self.save_results(output, f"{timestamp}.jbl")

            except Exception as e:
                print(f"❌ Error processing sequence_id {item.get('sequence_id', 'unknown')}: {e}")
                continue

        # Save final results
        self.save_results(output)
        print(f"✅ Processing complete! Processed {len(output)} queries.")
