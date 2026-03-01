"""
Subtask 1A Test Data Submission Generator
=======================================

This script generates submissions for Subtask 1A: Identification of Intended Ayahs and Hadiths.
It uses a fine-tuned model to identify spans of Quranic verses and Hadiths in LLM responses.
"""

import json
import os
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import dotenv_values
import time
import json_repair
import re

# Import ArabicTextNormalizer from dataset creation script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Copy ArabicTextNormalizer class from 02 script
class ArabicTextNormalizer:
    """Handle Arabic text normalization and tashkeel manipulation"""
    
    @staticmethod
    def remove_tashkeel(text: str) -> str:
        """Remove all Arabic diacritics (tashkeel)"""
        arabic_diacritics = 'ًٌٍَُِّْ'
        for diacritic in arabic_diacritics:
            text = text.replace(diacritic, '')
        return text
    
    @staticmethod
    def normalize_arabic(text: str) -> str:
        """Apply common Arabic normalization"""
        # Normalize Ya
        text = text.replace('ى', 'ي')
        # Normalize Alef variations
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        # Normalize Waw and Ya with hamza
        text = text.replace('ؤ', 'و').replace('ئ', 'ي')
        # Normalize Taa Marbouta in some contexts
        text = re.sub(r'ة(?=\s|$)', 'ه', text)
        return text


class Subtask1ASubmissionGenerator:
    """Generate Subtask 1A submissions using fine-tuned model"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        if api_key is None:
            # Try to load from .env file
            config = dotenv_values("../../.env")
            api_key = config.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.normalizer = ArabicTextNormalizer()
    
    def load_test_data(self, test_file_path: str) -> List[Dict[str, Any]]:
        """Load test data from JSONL file"""
        test_data = []
        with open(test_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
        return test_data
    
    def extract_spans_with_model(self, model_id: str, text: str) -> str:
        """Use fine-tuned model to extract Ayah and Hadith spans"""
        system_prompt = "Extract Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from the given text. Return only the exact quoted religious content."
        
        # Normalize Arabic text before processing
        normalized_text = self.normalizer.normalize_arabic(text)
        
        try:
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": normalized_text}
                ],
                temperature=0,
                max_tokens=3000  # Increased from 1000 to handle longer responses
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return ""
    
    def parse_model_output(self, model_output: str) -> List[Dict[str, Any]]:
        """Parse model output to extract detected phrases with spans"""
        try:
            # Try to parse as JSON first
            if model_output.strip().startswith('{'):
                parsed = json_repair.loads(model_output)
                if 'detected_phrases' in parsed:
                    return parsed['detected_phrases']
            
            # # Fallback: extract phrases using patterns (without spans)
            # phrases = []
            
            # # Look for Quranic verses in quotes or braces
            # ayah_patterns = [
            #     r'[{"]([^}"]*(?:قال|تعالى|الله)[^}"]*)[}"]',
            #     r'"([^"]*(?:يا أيها|إن|قل|والله)[^"]*)"',
            #     r'{([^}]*(?:الله|تعالى|سبحانه)[^}]*)}',
            # ]
            
            # for pattern in ayah_patterns:
            #     matches = re.finditer(pattern, model_output)
            #     for match in matches:
            #         phrases.append({
            #             "label": "Ayah",
            #             "value": match.group(1).strip()
            #         })
            
            # # Look for Hadith patterns
            # hadith_patterns = [
            #     r'قال رسول الله[^"]*"([^"]+)"',
            #     r'عن [^:]+:[^"]*"([^"]+)"',
            #     r'صلى الله عليه وسلم[^"]*"([^"]+)"',
            # ]
            
            # for pattern in hadith_patterns:
            #     matches = re.finditer(pattern, model_output)
            #     for match in matches:
            #         phrases.append({
            #             "label": "Hadith", 
            #             "value": match.group(1).strip()
            #         })
            
            # return phrases
            
        except Exception as e:
            print(f"Error parsing model output: {e}")
            return []
    
    def find_spans_in_text(self, text: str, phrases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find character spans of detected phrases in the original text with proper ordering"""
        spans = []
        
        # Sort phrases by span_order if available
        sorted_phrases = sorted(phrases, key=lambda x: x.get("span_order", 0))
        
        search_start = 0  # Track where to start searching next
        
        for phrase in sorted_phrases:
            phrase_text = phrase["value"]
            phrase_label = phrase["label"]
            
            # Try exact match first
            pos = text.find(phrase_text, search_start)
            found = False
            
            if pos != -1:
                spans.append({
                    "start": pos,
                    "end": pos + len(phrase_text) - 1,  # Inclusive end
                    "type": phrase_label,
                    "value": phrase_text
                })
                search_start = pos + len(phrase_text)
                found = True
            
            # If exact match not found, try with normalized text
            if not found:
                normalized_text = self.normalizer.normalize_arabic(text)
                normalized_phrase = self.normalizer.normalize_arabic(phrase_text)
                
                pos = normalized_text.find(normalized_phrase, search_start)
                if pos != -1:
                    spans.append({
                        "start": pos,
                        "end": pos + len(normalized_phrase) - 1,  # Inclusive end
                        "type": phrase_label,
                        "value": phrase_text
                    })
                    search_start = pos + len(normalized_phrase)
        
        return spans
    
    def generate_submission(self, model_id: str, test_file_path: str, output_file: str, max_samples: Optional[int] = None, verbose: bool = False) -> None:
        """Generate Subtask 1A submission file"""
        print(f"Starting Subtask 1A submission generation...")
        print(f"Model: {model_id}")
        print(f"Test file: {test_file_path}")
        print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
        
        # Load test data
        test_data = self.load_test_data(test_file_path)
        
        # Limit samples if max_samples is specified
        if max_samples is not None and max_samples > 0:
            test_data = test_data[:max_samples]
            print(f"Limiting to {max_samples} samples (out of {len(self.load_test_data(test_file_path))} total)")
        
        print(f"Processing {len(test_data)} test examples")
        
        # Prepare submission data
        submission_rows = []
        
        for i, example in enumerate(test_data):
            print(f"Processing example {i+1}/{len(test_data)}: {example['ID']}")
            
            response_text = example["Response"]
            question_id = example["ID"]
            
            # Get model prediction
            model_output = self.extract_spans_with_model(model_id, response_text)
            
            # Parse model output to get detected phrases
            detected_phrases = self.parse_model_output(model_output)
            
            if detected_phrases:
                try:
                    # Check if phrases already have span information
                    if detected_phrases and len(detected_phrases) > 0 and isinstance(detected_phrases[0], dict) and 'span_start' in detected_phrases[0] and 'span_end' in detected_phrases[0]:
                        # Recalculate spans using the value text to ensure accuracy
                        spans = self.find_spans_in_text(response_text, detected_phrases)
                    else:
                        # Find spans in original text (fallback for old format)
                        spans = self.find_spans_in_text(response_text, detected_phrases)
                except Exception as e:
                    print(f"Error processing detected phrases for {question_id}: {e}")
                    print(f"Model output: {model_output}")
                    print(f"Detected phrases: {detected_phrases}")
                    # Continue with empty spans to avoid breaking the loop
                    spans = []
                
                # Process spans if we have any
                if spans:
                    # print("spans:", spans)
                    # Remove duplicates based on start, end, and type
                    unique_spans = []
                    seen_spans = set()
                    
                    for span in spans:
                        span_key = (span["start"], span["end"], span["type"], span.get("value", ""))
                        if span_key not in seen_spans:
                            seen_spans.add(span_key)
                            unique_spans.append(span)
                    
                    for span in unique_spans:
                        row = {
                            "Question_ID": question_id,
                            "Span_Start": span["start"],
                            "Span_End": span["end"],  # Keep as-is (model provides inclusive end)
                            "Span_Type": span["type"],
                            
                        }
                        
                        # Add Original_Span column in verbose mode
                        if verbose:
                            # Extract the actual text using the span indices
                            # For inclusive end, use span["end"]+1 for slicing
                            original_span = response_text[span["start"]:span["end"]+1]
                            row["Original_Span"] = original_span
                            row["LLM_Value"] = span.get("value", "")
                        
                        submission_rows.append(row)
                else:
                    # No valid spans found, add No_Spans entry
                    row = {
                        "Question_ID": question_id,
                        "Span_Start": 0,
                        "Span_End": 0,
                        "Span_Type": "No_Spans"
                    }
                    
                    # Add empty columns in verbose mode
                    if verbose:
                        row["Original_Span"] = ""
                        row["LLM_Value"] = ""
                    
                    submission_rows.append(row)
            else:
                # No spans found
                row = {
                    "Question_ID": question_id,
                    "Span_Start": 0,
                    "Span_End": 0,
                    "Span_Type": "No_Spans"
                }
                
                # Add empty columns in verbose mode
                if verbose:
                    row["Original_Span"] = ""
                    row["LLM_Value"] = ""
                
                submission_rows.append(row)
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        # Create submission DataFrame and save as TSV
        df = pd.DataFrame(submission_rows)
        df.to_csv(output_file, sep='\t', index=False)
        
        print(f"\nSubmission generated successfully!")
        print(f"Output file: {output_file}")
        print(f"Total rows: {len(submission_rows)}")
        
        # Print summary statistics
        span_types = df['Span_Type'].value_counts()
        print(f"\nSpan type distribution:")
        for span_type, count in span_types.items():
            print(f"  {span_type}: {count}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Subtask 1A submission")
    parser.add_argument("model_id", help="Fine-tuned model ID")
    parser.add_argument("--test_file", 
                       default="../../datasets/taskA-testdata/Test_Subtask_1A/test_Subtask_1A.jsonl",
                       help="Path to test data file")
    parser.add_argument("--output", 
                       default="subtask1A_submission.tsv",
                       help="Output submission file path")
    parser.add_argument("--max-samples", 
                       type=int,
                       help="Maximum number of samples to process (for testing/debugging)")
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Include Original_Span column with extracted text for validation")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        return
    
    try:
        generator = Subtask1ASubmissionGenerator()
        generator.generate_submission(args.model_id, args.test_file, args.output, args.max_samples, args.verbose)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
