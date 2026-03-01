#!/usr/bin/env python3
"""
Subtask C: Correction of Erroneous Content for Ayahs and Hadiths
Generates competition submission file with corrected versions or 'خطأ' if no correction exists.
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


class SubtaskCCorrector:
    """Corrects detected Ayahs and Hadiths using search + LLM"""
    
    def __init__(self, api_key: Optional[str] = None, book_ids: List[str] = None):
        """Initialize corrector with search engine and LLM client"""
        # Initialize OpenAI client
        if api_key is None:
            config = dotenv_values("../.env")
            api_key = config.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4.1"
        
        # Set default book IDs for hadith filtering
        self.book_ids = book_ids if book_ids is not None else ["1.0", "2.0"]
        
        # Initialize search engine
        self.searcher = ReligiousTextSearcher(indices_dir="../datasets/indices")
        if not self.searcher.load_indices():
            raise RuntimeError("Failed to load search indices")
        
        # Rate limiting
        self.request_delay = 1
        
        # Store original test data for span extraction
        self.test_data = {}
        
    def load_test_data_jsonl(self, test_file_path: str):
        """Load original test data from JSONL format"""
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.test_data[data['ID']] = data['Response']
    
    def correct_phrase(self, phrase_text: str, span_type: str, max_results: int = 20) -> tuple:
        """Correct a single phrase and return (correction, search_results_texts)"""
        
        # Search for similar texts in appropriate corpus
        search_type = 'ayah' if span_type == 'Ayah' else 'hadith'
        results, _ = self.searcher.search_hierarchical(
            phrase_text, 
            search_type=search_type, 
            max_results=max_results,
            book_ids=self.book_ids
        )
        
        # Extract search result texts for verbose output
        search_texts = [result.original_text for result in results]
        
        # Generate LLM prompt with search results
        prompt = self._generate_correction_prompt(phrase_text, span_type, results)

        # print("*" * 20)
        # print(prompt)
        # print("*" * 20)
        
        # Call LLM for correction
        correction = self._call_llm_for_correction(prompt)
        
        return correction, search_texts
    
    def _generate_correction_prompt(self, phrase_text: str, span_type: str, search_results: List) -> str:
        """Generate correction prompt for LLM with enhanced search context"""
        
        # Build search results context with match types and confidence scores
        results_text = ""
        if search_results:
            for i, result in enumerate(search_results, 1):
                if span_type == 'Ayah':
                    source = f"Surah {result.surah_name}:{result.ayah_id}"
                else:
                    source = f"Book {result.book_id}"
                    if result.hadith_title:
                        source += f" ({result.hadith_title[:60]}...)"
                
                # Include match type and confidence for better LLM understanding
                match_info = "" #f"[{result.match_type.upper()}, conf: {result.confidence:.2f}]"
                results_text += f"\n===== Hadith : ({i})=====\n{match_info} [{source}] \"{result.original_text}\""
        else:
            results_text = "\n(No database matches found)"


        return f"""
You are a highly specialized AI assistant trained to validate the accuracy of Islamic texts, specifically Quranic verses (Ayahs) and Prophetic sayings (Hadiths). Your task is to correct or identify as incorrect a given `$original_text` based on a provided list of `$database_recommended_texts`.

**Instructions:**
* **Validation:** Carefully examine the `$original_text`.
* **Correction:** Compare the `$original_text` against the `$database_recommended_texts` to find a perfect, exact match. An exact match must include all diacritics and characters.
* **Output:** Respond with a single JSON object containing the key `"Correction"`. The value should be the full, corrected text or `"خطأ"` if no correction is found.

---

### **For Ayahs (Quranic Verses)**

* **Correction Criteria:** A corrected Ayah must be a complete and accurate verse from the Quran, including all diacritics. Partial quotes or paraphrased versions are considered incorrect.
* **Output Logic:**
    * If the `$original_text` is an exact match to a text in `$database_recommended_texts`, provide the full Ayah or Hadith text as the correction.
    * If the `$original_text` is incorrect but a complete and accurate version of the Ayah exists in `$database_recommended_texts`, provide that full text as the correction.
    * The correction must be the full Ayah (or more) text, including proper diacritics and formatting.
    * This generation will be evaluated via Word-Error Rate (WER) and BLEU score, so ensure the correction is as close to the original Ayah as possible.

**Example: Ayah Correction**
* `$original_text`: "قُل لَّا يَعْلَمُ مَن فِي السَّمَاوَاتِ وَالْأَرْضِ الْغَيْبَ إِلَّا اللَّهُ"
* `$database_recommended_texts`: `["قُل لَّا يَعْلَمُ مَن فِي السَّمَاوَاتِ وَالْأَرْضِ الْغَيْبَ إِلَّا اللَّهُ", "..."]`
* **Your Output:** `{{"Correction": "قُل لَّا يَعْلَمُ مَن فِي السَّمَاوَاتِ وَالْأَرْضِ الْغَيْبَ إِلَّا اللَّهُ"}}`

Common introductory phrases to IGNORE when correcting:
    - وقال / قال (and he said / he said)
    - روى البخاري / روى مسلم (narrated by Bukhari/Muslim)
    - عن الرسول / عن النبي (from the Prophet)
    - أنه قال / انه قال (that he said)
    - صلى الله عليه وسلم (peace be upon him)
    - رضي الله عنه (may Allah be pleased with him)
    - يقول الله تعالى (Allah says)
    - في كتابه العزيز (in His noble book)
    - Similar narrative/attribution phrases

---

### **For Hadiths (Prophetic Sayings)**

* **Correction Criteria:** A corrected Hadith must be a complete and accurate saying from an established Islamic source.
* **Output Logic:**
    * If the `$original_text` is an exact match to a text in `$database_recommended_texts`, provide the text itself as the correction.
    * If the `$original_text` is incorrect and a complete, correct version exists in `$database_recommended_texts`, provide that full text as the correction.
    * If the `$original_text` is incorrect and you cannot find any suitable, correct version in the `$database_recommended_texts`, the `Correction` must be the word `"خطأ"`.

**Example 1: Hadith Correction**
* `$original_text`: "مفاتح الغيب خمس لا يعلمهن إلا الله"
* `$database_recommended_texts`: `["مفاتح الغيب خمس لا يعلمها إلا الله", "..."]`
* **Your Output:** `{{"Correction": "مفاتح الغيب خمس لا يعلمها إلا الله"}}`

**Example 2: No Correct Hadith Found**
* `$original_text`: "وَمَا كَانَ لِسْلَيْمَنَ نَفَقَةُ وَلَا هُوَ يَعْلَمُ مَا فِيَ غَيْبِ السَّمُوُتِ وَالْأَرْض وَٱللَّهُ عَلِمُ بِمَا تَعْمَلُونَ"
* `$database_recommended_texts`: `["وَمَا كَانَ لِسْلَيْمَنَ نَفَقَةُ", "وَإِنَّا لَجَاعِلُونَ مَا عَلَيْهَا صَعِيدًا جُرُزًا", "..."]`
* **Your Output:** `{{"Correction": "خطأ"}}`

---

**Respond JSON only: `{{"Correction": "complete_correct_text"}}` or `{{"Correction": "خطأ"}}`**

**Actual Task Inputs:**
# Original Text to be validated:
"{phrase_text}"


# database recommended texts
{results_text}

Respond JSON only: {{"Correction": "complete_correct_text"}} or {{"Correction": "خطأ"}}
"""
        
        # Common introductory phrases to ignore
        intro_phrases = """Common introductory phrases to IGNORE when correcting:
    - وقال / قال (and he said / he said)
    - روى البخاري / روى مسلم (narrated by Bukhari/Muslim)
    - عن الرسول / عن النبي (from the Prophet)
    - أنه قال / انه قال (that he said)
    - صلى الله عليه وسلم (peace be upon him)
    - رضي الله عنه (may Allah be pleased with him)
    - يقول الله تعالى (Allah says)
    - في كتابه العزيز (in His noble book)
    - Similar narrative/attribution phrases"""
        
        # Correction criteria based on text type
        if span_type == 'Ayah':
            criteria = f"""{intro_phrases}
    
    Quranic correction rules:
    - FOCUS on the actual Quranic text, IGNORE narrative introductions
    - Find the closest complete Quranic verse(s) that match the intended meaning
    - Provide the complete authentic verse(s) from the Quran
    - Include proper diacritics and formatting
    - If partial verse is given, give the full verse as is but with proper correction (if required)
    - If the text seems to be a mix of verses, provide the correct complete verses
    - remove surah name from the verse text (if exists).
    - Preserve verse numbers if they help clarify the correction"""
        else:
            criteria = f"""{intro_phrases}
    
    Hadith correction rules:
    - FOCUS on the actual Hadith text, IGNORE chain of narration
    - Study the meaning of Hadith text, if it closer (in meaning) to a given Database match , use the matched Database matche to be the correction
    - Find the closest complete authentic Hadith that matches the intended meaning
    - If partial Hadith is given, return the full Hadith text.
    - Include proper diacritics and formatting
    - Chain variations (Bukhari/Muslim/etc.) should be ignored in correction"""
        
        prompt = f"""
Task: Provide the correct complete authentic Islamic text for the given incorrect {span_type.lower()}.

There's not always a correctness for every phrase, so you must determine if a correction exists.
correction in Verse mean high similar text match from database, while in Hadith it could be high similar text or meaning match.
Maybe the text is acceptable as is or not relevant so far to any of search results or any religion text (Especially Quranic verses).

text to be validated: "{phrase_text}"

Database matches:{results_text}

{criteria}

CORRECTION APPROACH:
1. Strip away any introductory/narrative phrases from the incorrect text
2. Identify the core religious content that needs correction
3. Find the most similar authentic text from the database matches
4. Prioritize results with higher confidence scores and exact/embedding matches
5. Consider partial matches if they have strong semantic similarity


OUTPUT FORMAT:
- If you find a matching authentic text: Return the complete correct text
- If no similar authentic text exists: Return exactly "خطأ"

CRITICAL: Only provide corrections for texts that have clear authentic equivalents. If uncertain, return "خطأ".

Use your background knowledge about Islamic texts to ensure that the `Correction` value is accurate, whether the suggested texts are relevant or not correct.

Respond JSON only: {{"Correction": "complete_correct_text"}} or {{"Correction": "خطأ"}}"""
        
        return prompt
    
    def _call_llm_for_correction(self, prompt: str) -> str:
        """Call LLM and return correction result"""
        try:
            messages = [
                {"role": "system", "content": "You are an expert Islamic scholar specializing in Quranic verses and Hadith authentication and correction."},
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
            
            correction = result.get("Correction", "خطأ")
            return correction
            
        except Exception as e:
            print(f"LLM correction error: {e}")
            return "خطأ"  # Default to error on exceptions
    
    def process_phrases_file(self, input_file: str, output_file: str, test_data_file: str, max_results: int = 10, verbose: bool = False, sequence_id_filter: int = None):
        """Process all phrases in the TSV file with progressive saving"""
        # Load test data for span extraction
        self.load_test_data_jsonl(test_data_file)
        
        df = pd.read_csv(input_file, sep='\t')
        
        # Filter to specific sequence ID if provided
        if sequence_id_filter is not None:
            if sequence_id_filter < 1 or sequence_id_filter > len(df):
                raise ValueError(f"Sequence ID {sequence_id_filter} out of range (1-{len(df)})")
            df = df.iloc[[sequence_id_filter - 1]]  # Convert to 0-based index
            print(f"🔍 Debug mode: Processing only Sequence_ID {sequence_id_filter}")
        
        print(f"Processing {len(df)} phrases...")
        print(f"Results will be saved progressively to: {output_file}")
        if verbose:
            print("Verbose mode: Including debug columns")
        
        # Initialize counters
        corrected_count = 0
        error_count = 0
        
        # Open output file for progressive writing
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write headers based on mode
            if verbose:
                f.write("Sequence_ID\tCorrection\tOriginal_Span\tSpan_Type\tTop_Found_Results\n")
            else:
                f.write("Sequence_ID\tCorrection\n")
            
            for idx, row in df.iterrows():
                sequence_id = sequence_id_filter if sequence_id_filter is not None else idx + 1
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
                
                print(f"  {sequence_id}/{len(df)}: Correcting {span_type} - {question_id}")
                
                # Correct phrase using extracted span
                correction, search_texts = self.correct_phrase(actual_span, span_type, max_results)
                
                # Get results for verbose logging
                if verbose:
                    results, _ = self.searcher.search_hierarchical(actual_span, 'ayah' if span_type == 'Ayah' else 'hadith', max_results=3, book_ids=self.book_ids)
                    if results:
                        match_types = [r.match_type for r in results[:3]]
                        has_reranked = any('reranked' in mt for mt in match_types)
                        rerank_info = " (🔄 reranked)" if has_reranked else ""
                        print(f"    Found via: {', '.join(match_types)}{rerank_info}")
                
                # Write result with appropriate format
                if verbose:
                    # Escape tabs and newlines in text fields
                    clean_span = actual_span.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                    clean_correction = correction.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                    search_results_str = " | ".join([text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ') for text in search_texts])
                    f.write(f"{sequence_id}\t{clean_correction}\t{clean_span}\t{span_type}\t{search_results_str}\n")
                else:
                    # Competition submission format
                    f.write(f"{sequence_id}\t{correction}\n")
                
                f.flush()  # Ensure data is written to disk
                
                # Update counters
                if correction == 'خطأ':
                    error_count += 1
                else:
                    corrected_count += 1
                
                # Print progress every 10 items
                if sequence_id % 10 == 0 or sequence_id == len(df):
                    print(f"    Progress: {sequence_id}/{len(df)} | Corrected: {corrected_count} | Errors: {error_count}")
        
        print(f"\nProcessing complete!")
        print(f"Submission file saved: {output_file}")
        print(f"Total phrases processed: {len(df)}")
        print(f"  Corrected: {corrected_count}")
        print(f"  Errors (خطأ): {error_count}")


def main():
    parser = argparse.ArgumentParser(description='Subtask C: Correct Ayah/Hadith content')
    parser.add_argument('--phrases-detected-tsv', required=True, 
                       help='Input TSV file with detected phrases (from Subtask A)')
    parser.add_argument('--test-data-file', required=True,
                       help='Original test data JSONL file for span extraction')
    parser.add_argument('--output_file_name', required=True,
                       help='Output submission file name')
    parser.add_argument('--max-search-results', type=int, default=20,
                       help='Maximum search results to use for correction')
    parser.add_argument('--book-ids', nargs='+', default=["1.0", "2.0"],
                       help='Book IDs to filter hadith results (default: 1.0 2.0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Include debug columns: Original_Span, Span_Type, Top_Found_Results')
    parser.add_argument('--sequence-id', type=int,
                       help='Process only specific Sequence_ID for debugging (1-based index)')
    
    args = parser.parse_args()
    
    # Initialize corrector with book IDs
    corrector = SubtaskCCorrector(book_ids=args.book_ids)
    
    print(f"📚 Using hadith book IDs: {', '.join(args.book_ids)}")
    
    # Process phrases with progressive saving
    corrector.process_phrases_file(
        args.phrases_detected_tsv, 
        args.output_file_name,
        args.test_data_file,
        args.max_search_results,
        args.verbose,
        args.sequence_id
    )


if __name__ == "__main__":
    main()
