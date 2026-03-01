"""
Fine-tuned Model Validation Script
=================================

This script validates a fine-tuned OpenAI model by running predictions on a dataset
and optionally using another model as a judge to evaluate the outputs.
"""

import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import dotenv_values
import time
import json_repair

# Import ArabicTextNormalizer from dataset creation script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

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
        import re
        text = re.sub(r'ة(?=\s|$)', 'ه', text)
        return text


class ModelValidator:
    """Validate fine-tuned model performance"""
    
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
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load validation dataset from .jsonl or .tsv file"""
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        elif dataset_path.endswith('.tsv'):
            import pandas as pd
            df = pd.read_csv(dataset_path, sep='\t')
            dataset = []
            for _, row in df.iterrows():
                # Convert TSV row to message format
                content = f'{{"detected_phrases": [{{"label": "{row["Label"]}", "value": "{row["Original_Span"]}"}}]}}'
                dataset.append({
                    "messages": [
                        {"role": "system", "content": "Extract Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from the given text. Return only the exact quoted religious content."},
                        {"role": "user", "content": row["Response"]},
                        {"role": "assistant", "content": content}
                    ]
                })
            return dataset
        else:
            raise ValueError("Dataset must be .jsonl or .tsv file")
    
    def run_prediction(self, model_id: str, messages: List[Dict[str, str]]) -> str:
        """Run prediction using fine-tuned model"""
        try:
            # Normalize Arabic text in user message
            normalized_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    normalized_content = self.normalizer.normalize_arabic(msg["content"])
                    normalized_messages.append({"role": msg["role"], "content": normalized_content})
                else:
                    normalized_messages.append(msg)
            
            response = self.client.chat.completions.create(
                model=model_id,
                messages=normalized_messages,
                temperature=0,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in prediction: {e}")
            return ""
    
    def judge_output(self, expected: str, predicted: str, input_text: str) -> Dict[str, Any]:
        """Use OpenAI to judge the quality of the prediction"""
        judge_prompt = f"""
You are evaluating the output of a model that extracts Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from Arabic text.

Input text: {input_text}

Expected output: {expected}

Model output: {predicted}

Evaluate if the model output correctly identifies and extracts the religious phrases. 

Respond with JSON in this format:
{{
    "status": "accepted" or "rejected",
    "comment": "Brief explanation if rejected, empty string if accepted"
}}

Criteria for acceptance:
- All Quranic verses and Hadiths are correctly identified
- No false positives (non-religious text marked as religious)
- Exact text extraction without modifications
- Proper labeling (Ayah vs Hadith)
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of Islamic text extraction models."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0,
                max_tokens=200
            )
            
            result = json_repair.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Error in judging: {e}")
            return {"status": "rejected", "comment": f"Judge error: {str(e)}"}
    
    def validate_model(self, model_id: str, dataset_path: str, use_judge: bool = False) -> Dict[str, Any]:
        """Validate the fine-tuned model"""
        print(f"Starting validation for model: {model_id}")
        print(f"Dataset: {dataset_path}")
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        print(f"Loaded {len(dataset)} examples")
        
        # Create output directory
        output_dir = f"output/validation/{model_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare results
        results = []
        scores = {
            "total_examples": len(dataset),
            "processed_examples": 0,
            "accepted": 0,
            "rejected": 0,
            "judge_enabled": use_judge,
            "model_id": model_id,
            "dataset_path": dataset_path,
            "timestamp": datetime.now().isoformat()
        }
        
        # Process each example
        for i, example in enumerate(dataset):
            print(f"Processing example {i+1}/{len(dataset)}...")
            
            # Extract input messages (excluding assistant response)
            messages = [msg for msg in example["messages"] if msg["role"] != "assistant"]
            expected_output = next(msg["content"] for msg in example["messages"] if msg["role"] == "assistant")
            input_text = next(msg["content"] for msg in example["messages"] if msg["role"] == "user")
            
            # Get model prediction
            predicted_output = self.run_prediction(model_id, messages)
            
            result = {
                "example_id": i,
                "input": input_text,
                "expected": expected_output,
                "predicted": predicted_output,
                "timestamp": datetime.now().isoformat()
            }
            
            # Judge output if requested
            if use_judge and predicted_output:
                judgment = self.judge_output(expected_output, predicted_output, input_text)
                result["judge_status"] = judgment["status"]
                result["judge_comment"] = judgment["comment"]
                
                if judgment["status"] == "accepted":
                    scores["accepted"] += 1
                else:
                    scores["rejected"] += 1
            
            results.append(result)
            scores["processed_examples"] += 1
            
            # Add small delay to avoid rate limits
            time.sleep(0.1)
        
        # Calculate final scores
        if use_judge:
            scores["accuracy"] = scores["accepted"] / scores["processed_examples"] if scores["processed_examples"] > 0 else 0
        
        # Save results
        output_file = os.path.join(output_dir, "llm-output.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        score_file = os.path.join(output_dir, "score.json")
        with open(score_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        
        print(f"\nValidation complete!")
        print(f"Results saved to: {output_file}")
        print(f"Scores saved to: {score_file}")
        
        if use_judge:
            print(f"Accuracy: {scores['accuracy']:.2%}")
            print(f"Accepted: {scores['accepted']}/{scores['processed_examples']}")
            print(f"Rejected: {scores['rejected']}/{scores['processed_examples']}")
        
        return scores


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Validate fine-tuned model")
    parser.add_argument("model_id", help="Fine-tuned model ID")
    parser.add_argument("dataset_path", help="Path to validation dataset (.jsonl)")
    parser.add_argument("--judge", action="store_true", help="Use OpenAI model as judge")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        return
    
    if not (args.dataset_path.endswith('.jsonl') or args.dataset_path.endswith('.tsv')):
        print("Error: Dataset must be a .jsonl or .tsv file")
        return
    
    try:
        validator = ModelValidator()
        validator.validate_model(args.model_id, args.dataset_path, args.judge)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
