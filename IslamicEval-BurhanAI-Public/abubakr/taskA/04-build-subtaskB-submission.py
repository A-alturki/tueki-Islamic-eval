#!/usr/bin/env python3
"""
Subtask B: Validation of content accuracy for Ayahs and Hadiths
Generates competition submission file with Correct/Incorrect labels.
"""

import argparse
import pandas as pd
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import dotenv_values
import json_repair

# Import the search engine
import sys
import importlib.util
sys.path.append('.')

# Load the search module
spec = importlib.util.spec_from_file_location("search_module", "02-search-religion-text.py")
search_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(search_module)
ReligiousTextSearcher = search_module.ReligiousTextSearcher


class SubtaskBValidator:
    """Validates detected Ayahs and Hadiths using search + LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize validator with search engine and LLM client"""
        # Initialize OpenAI client
        if api_key is None:
            config = dotenv_values("../.env")
            api_key = config.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4.1-mini"
        
        # Initialize search engine
        self.searcher = ReligiousTextSearcher(indices_dir="../datasets/indices")
        if not self.searcher.load_indices():
            raise RuntimeError("Failed to load search indices")
        
        # Rate limiting
        self.request_delay = 1
        
        # Store original test data for span extraction
        self.test_data = {}
        
    def load_test_data(self, test_file_path: str):
        """Load original test data to extract full spans"""
        import json
        
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.test_data[data['ID']] = data['Response']
        
    def validate_phrase(self, phrase_text: str, span_type: str, max_results: int = 7) -> tuple:
        """Validate a single phrase and return (label, search_results_texts)"""
        
        # Search for similar texts in appropriate corpus
        search_type = 'ayah' if span_type == 'Ayah' else 'hadith'
        results, _ = self.searcher.search_hierarchical(
            phrase_text, 
            search_type=search_type, 
            max_results=max_results
        )
        
        # Extract search result texts for verbose output
        search_texts = [result.original_text for result in results]
        
        # Generate LLM prompt with search results
        prompt = self._generate_validation_prompt(phrase_text, span_type, results)
        
        # Call LLM for validation
        label = self._call_llm_for_validation(prompt)
        
        return label, search_texts
    
    def _generate_validation_prompt(self, phrase_text: str, span_type: str, search_results: List) -> str:
        """Generate validation prompt for LLM with enhanced criteria"""
        
        # Build search results context
        results_text = ""
        if search_results:
            for i, result in enumerate(search_results, 1):
                source = f"Surah {result.surah_name}:{result.ayah_id}" if span_type == 'Ayah' else f"Book {result.book_id}"
                results_text += f"\n{i}. [{source}] \"{result.original_text}\""
        else:
            results_text = "\n(No database matches found)"
        
        # Common introductory phrases to ignore
        intro_phrases = """Common introductory phrases to IGNORE when validating:
    - وقال / قال (and he said / he said)
    - روى البخاري / روى مسلم (narrated by Bukhari/Muslim)
    - عن الرسول / عن النبي (from the Prophet)
    - أنه قال / انه قال (that he said)
    - صلى الله عليه وسلم (peace be upon him)
    - رضي الله عنه (may Allah be pleased with him)
    - يقول الله تعالى (Allah says)
    - في كتابه العزيز (in His noble book)
    - Similar narrative/attribution phrases"""
        
        # Validation criteria based on text type
        if span_type == 'Ayah':
            criteria = f"""{intro_phrases}
    
    Quranic validation rules:
    - FOCUS on the actual Quranic text, IGNORE narrative introductions
    - Accept partial verses if they match authentic Quranic text exactly
    - Accept complete verses or multiple consecutive verses
    - The core Quranic words must be accurate (ignore intro phrases)
    - Word order within the Quranic portion must be preserved
    - Diacritics should be reasonably accurate"""
        else:
            criteria = f"""{intro_phrases}
    
    Hadith validation rules:
    - FOCUS on the actual Hadith text, IGNORE chain of narration
    - Accept partial Hadith if it matches authentic text exactly
    - Accept complete Hadith text
    - The core Hadith words must be accurate (ignore intro phrases)
    - Meaning-preserving paraphrases are still INCORRECT
    - Chain variations (Bukhari/Muslim/etc.) are acceptable"""
        
        prompt = f"""You are an expert Islamic scholar specializing in Quranic verses and Hadith authentication.

Task: Validate if the detected {span_type.lower()} contains authentic Islamic text.

Detected text: "{phrase_text}"

Database matches:{results_text}

{criteria}

EVALUATION APPROACH:
1. Strip away any introductory/narrative phrases from the detected text
2. Focus on the core religious content (Quranic verses or Hadith text)
3. Check if this core content matches authentic sources exactly
4. Partial matches are CORRECT if the included portion is accurate

CRITICAL: If the core religious text (after removing introductions) matches authentic sources exactly, mark as Correct.

Respond JSON only: {{"Label": "Correct"}} or {{"Label": "Incorrect"}}"""
        
        return prompt
    
    def _call_llm_for_validation(self, prompt: str) -> str:
        """Call LLM and return validation result"""
        try:
            messages = [
                {"role": "system", "content": "You are an expert Islamic scholar. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            # Parse response
            llm_response = response.choices[0].message.content.strip()
            result = json_repair.loads(llm_response)
            
            label = result.get("Label", "Incorrect")
            return "Correct" if label == "Correct" else "Incorrect"
            
        except Exception as e:
            print(f"LLM validation error: {e}")
            return "Incorrect"  # Default to incorrect on errors
    
    def process_phrases_file(self, input_file: str, output_file: str, test_data_file: str, max_results: int = 7, verbose: bool = False):
        """Process all phrases in the TSV file with progressive saving"""
        # Load test data for span extraction
        self.load_test_data(test_data_file)
        
        df = pd.read_csv(input_file, sep='\t')
        
        print(f"Processing {len(df)} phrases...")
        print(f"Results will be saved progressively to: {output_file}")
        if verbose:
            print("Verbose mode: Including debug columns")
        
        # Initialize counters
        correct_count = 0
        incorrect_count = 0
        
        # Open output file for progressive writing
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write headers based on mode
            if verbose:
                f.write("Sequence_ID\tLabel\tOriginal_Span\tSpan_Type\tTop_Found_Results\n")
            else:
                f.write("Sequence_ID\tLabel\n")
            
            for idx, row in df.iterrows():
                sequence_id = idx + 1
                question_id = row['Question_ID']
                span_start = int(row['Span_Start'])
                span_end = int(row['Span_End'])
                span_type = row['Span_Type']
                
                # Extract full span from original text
                if question_id in self.test_data:
                    full_text = self.test_data[question_id]
                    # Extract actual span text (end is inclusive in competition format)
                    actual_span = full_text[span_start:span_end + 1]
                else:
                    print(f"  Warning: Question {question_id} not found in test data")
                    actual_span = row.get('Original_Span', '')  # Fallback to truncated span
                
                print(f"  {sequence_id}/{len(df)}: Validating {span_type} - {question_id}")
                
                # Validate phrase using extracted span
                label, search_texts = self.validate_phrase(actual_span, span_type, max_results)
                
                # Write result with appropriate format
                if verbose:
                    # Escape tabs and newlines in text fields
                    clean_span = actual_span.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                    search_results_str = " | ".join([text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ') for text in search_texts])
                    f.write(f"{sequence_id}\t{label}\t{clean_span}\t{span_type}\t{search_results_str}\n")
                else:
                    # Minimal submission format (no headers)
                    f.write(f"{sequence_id}\t{label}\n")
                
                f.flush()  # Ensure data is written to disk
                
                # Update counters
                if label == 'Correct':
                    correct_count += 1
                else:
                    incorrect_count += 1
                
                # Print progress every 10 items
                if sequence_id % 10 == 0 or sequence_id == len(df):
                    print(f"    Progress: {sequence_id}/{len(df)} | Correct: {correct_count} | Incorrect: {incorrect_count}")
        
        print(f"\nProcessing complete!")
        print(f"Submission file saved: {output_file}")
        print(f"Total phrases processed: {len(df)}")
        print(f"  Correct: {correct_count}")
        print(f"  Incorrect: {incorrect_count}")
    
    def save_submission_file(self, results: List[Dict], output_file: str):
        """Save results in competition submission format (DEPRECATED - use progressive saving)"""
        print("Warning: This method is deprecated. Results are now saved progressively during processing.")
        pass


def main():
    parser = argparse.ArgumentParser(description='Subtask B: Validate Ayah/Hadith accuracy')
    parser.add_argument('--phrases-detected-tsv', required=True, 
                       help='Input TSV file with detected phrases')
    parser.add_argument('--test-data-file', required=True,
                       help='Original test data JSONL file for span extraction')
    parser.add_argument('--output_file_name', required=True,
                       help='Output submission file name')
    parser.add_argument('--max-search-results', type=int, default=7,
                       help='Maximum search results to use for validation')
    parser.add_argument('--verbose', action='store_true',
                       help='Include debug columns: Original_Span, Span_Type, Top_Found_Results')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = SubtaskBValidator()
    
    # Process phrases with progressive saving
    validator.process_phrases_file(
        args.phrases_detected_tsv, 
        args.output_file_name,
        args.test_data_file,
        args.max_search_results,
        args.verbose
    )


if __name__ == "__main__":
    main()
