"""
Phrase Detection Fine-tuning Task
=================================

This script handles the fine-tuning process for the phrase detection model
using OpenAI's fine-tuning API. It includes data validation, model training,
and evaluation utilities.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI
from dotenv import dotenv_values
import pandas as pd


class PhraseDetectionFineTuner:
    """Handle OpenAI fine-tuning for phrase detection task"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        if api_key is None:
            # Try to load from .env file
            config = dotenv_values("../../.env")
            api_key = config.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.fine_tune_job = None
    
    def validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Validate the dataset format and content"""
        print("Validating dataset...")
        
        validation_results = {
            "total_examples": 0,
            "valid_examples": 0,
            "errors": [],
            "statistics": {
                "ayah_examples": 0,
                "hadith_examples": 0,
                "negative_examples": 0,
                "multi_phrase_examples": 0
            }
        }
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    validation_results["total_examples"] += 1
                    
                    try:
                        # Parse JSON
                        example = json.loads(line.strip())
                        
                        # Validate structure
                        if not self._validate_example_structure(example):
                            validation_results["errors"].append(
                                f"Line {line_num}: Invalid structure"
                            )
                            continue
                        
                        # Collect statistics
                        self._update_statistics(example, validation_results["statistics"])
                        validation_results["valid_examples"] += 1
                        
                    except json.JSONDecodeError as e:
                        validation_results["errors"].append(
                            f"Line {line_num}: JSON decode error - {str(e)}"
                        )
                    except Exception as e:
                        validation_results["errors"].append(
                            f"Line {line_num}: Validation error - {str(e)}"
                        )
        
        except FileNotFoundError:
            validation_results["errors"].append(f"Dataset file not found: {dataset_path}")
        
        # Print validation summary
        self._print_validation_summary(validation_results)
        
        return validation_results
    
    def _validate_example_structure(self, example: Dict[str, Any]) -> bool:
        """Validate individual example structure"""
        # Check required fields
        if "messages" not in example:
            return False
        
        messages = example["messages"]
        if len(messages) != 3:
            return False
        
        # Check message roles
        expected_roles = ["system", "user", "assistant"]
        for i, msg in enumerate(messages):
            if msg.get("role") != expected_roles[i]:
                return False
            if "content" not in msg:
                return False
        
        # Validate assistant response is valid JSON
        try:
            assistant_content = json.loads(messages[2]["content"])
            if "detected_phrases" not in assistant_content:
                return False
            
            # Validate phrase structure
            for phrase in assistant_content["detected_phrases"]:
                if not isinstance(phrase, dict):
                    return False
                if "label" not in phrase or "value" not in phrase:
                    return False
                if phrase["label"] not in ["Ayah", "Hadith"]:
                    return False
        
        except json.JSONDecodeError:
            return False
        
        return True
    
    def _update_statistics(self, example: Dict[str, Any], stats: Dict[str, int]) -> None:
        """Update statistics based on example content"""
        assistant_content = json.loads(example["messages"][2]["content"])
        phrases = assistant_content["detected_phrases"]
        
        if not phrases:
            stats["negative_examples"] += 1
        elif len(phrases) > 1:
            stats["multi_phrase_examples"] += 1
        
        for phrase in phrases:
            if phrase["label"] == "Ayah":
                stats["ayah_examples"] += 1
            elif phrase["label"] == "Hadith":
                stats["hadith_examples"] += 1
    
    def _print_validation_summary(self, results: Dict[str, Any]) -> None:
        """Print validation summary"""
        print(f"\n=== Dataset Validation Summary ===")
        print(f"Total examples: {results['total_examples']}")
        print(f"Valid examples: {results['valid_examples']}")
        print(f"Invalid examples: {results['total_examples'] - results['valid_examples']}")
        
        if results['errors']:
            print(f"\nErrors found: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
        
        print(f"\n=== Content Statistics ===")
        stats = results['statistics']
        print(f"Ayah examples: {stats['ayah_examples']}")
        print(f"Hadith examples: {stats['hadith_examples']}")
        print(f"Negative examples: {stats['negative_examples']}")
        print(f"Multi-phrase examples: {stats['multi_phrase_examples']}")
    
    def upload_dataset(self, dataset_path: str) -> str:
        """Upload dataset to OpenAI for fine-tuning"""
        print("Uploading dataset to OpenAI...")
        
        try:
            with open(dataset_path, 'rb') as f:
                response = self.client.files.create(
                    file=f,
                    purpose='fine-tune'
                )
            
            file_id = response.id
            print(f"Dataset uploaded successfully. File ID: {file_id}")
            return file_id
        
        except Exception as e:
            print(f"Error uploading dataset: {str(e)}")
            raise
    
    def start_fine_tuning(self, 
                         file_id: str, 
                         model: str = "gpt-3.5-turbo-1106",
                         n_epochs: int = 3,
                         learning_rate_multiplier: float = 2.0,
                         batch_size: int = 1) -> str:
        """Start the fine-tuning job"""
        print(f"Starting fine-tuning with model: {model}")
        
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=model,
                hyperparameters={
                    "n_epochs": n_epochs,
                    "learning_rate_multiplier": learning_rate_multiplier,
                    "batch_size": batch_size
                },
                suffix="phrase-detection"  # Will be added to model name
            )
            
            job_id = response.id
            self.fine_tune_job = job_id
            print(f"Fine-tuning job started. Job ID: {job_id}")
            return job_id
        
        except Exception as e:
            print(f"Error starting fine-tuning: {str(e)}")
            raise
    
    def monitor_fine_tuning(self, job_id: str) -> None:
        """Monitor the fine-tuning progress"""
        print("Monitoring fine-tuning progress...")
        
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                status = job.status
                
                print(f"Status: {status}")
                
                if status == "succeeded":
                    print(f"Fine-tuning completed successfully!")
                    print(f"Fine-tuned model ID: {job.fine_tuned_model}")
                    break
                elif status == "failed":
                    print(f"Fine-tuning failed: {job.error}")
                    break
                elif status == "cancelled":
                    print("Fine-tuning was cancelled")
                    break
                else:
                    # Still running, wait and check again
                    print("Fine-tuning in progress... checking again in 60 seconds")
                    time.sleep(60)
            
            except Exception as e:
                print(f"Error checking fine-tuning status: {str(e)}")
                time.sleep(60)
    
    def test_model(self, model_id: str, test_cases: List[str]) -> None:
        """Test the fine-tuned model with sample inputs"""
        print(f"Testing model: {model_id}")
        
        system_prompt = "Extract Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from the given text. Return only the exact quoted religious content."
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Input: {test_text}")
            
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": test_text}
                    ],
                    max_tokens=500,
                    temperature=0
                )
                
                result = response.choices[0].message.content
                print(f"Output: {result}")
                
                # Try to parse as JSON for better formatting
                try:
                    parsed_result = json.loads(result)
                    print(f"Parsed: {json.dumps(parsed_result, ensure_ascii=False, indent=2)}")
                except json.JSONDecodeError:
                    pass
            
            except Exception as e:
                print(f"Error testing model: {str(e)}")
    
    def evaluate_model(self, model_id: str, test_dataset_path: str) -> Dict[str, float]:
        """Evaluate model performance on test dataset"""
        print("Evaluating model performance...")
        
        total_examples = 0
        correct_predictions = 0
        
        system_prompt = "Extract Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from the given text. Return only the exact quoted religious content."
        
        try:
            with open(test_dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line.strip())
                    user_input = example["messages"][1]["content"]
                    expected_output = example["messages"][2]["content"]
                    
                    # Get model prediction
                    try:
                        response = self.client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_input}
                            ],
                            max_tokens=500,
                            temperature=0
                        )
                        
                        predicted_output = response.choices[0].message.content
                        
                        # Simple accuracy check (can be improved)
                        if self._compare_outputs(predicted_output, expected_output):
                            correct_predictions += 1
                        
                        total_examples += 1
                        
                        if total_examples % 10 == 0:
                            print(f"Evaluated {total_examples} examples...")
                    
                    except Exception as e:
                        print(f"Error evaluating example: {str(e)}")
                        continue
        
        except FileNotFoundError:
            print(f"Test dataset not found: {test_dataset_path}")
            return {}
        
        accuracy = correct_predictions / total_examples if total_examples > 0 else 0
        
        results = {
            "accuracy": accuracy,
            "total_examples": total_examples,
            "correct_predictions": correct_predictions
        }
        
        print(f"\n=== Evaluation Results ===")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Correct predictions: {correct_predictions}/{total_examples}")
        
        return results
    
    def _compare_outputs(self, predicted: str, expected: str) -> bool:
        """Compare predicted and expected outputs (simplified)"""
        try:
            pred_json = json.loads(predicted)
            exp_json = json.loads(expected)
            
            # Simple comparison - can be made more sophisticated
            return pred_json == exp_json
        
        except json.JSONDecodeError:
            return predicted.strip() == expected.strip()
    
    def run_full_pipeline(self, 
                         dataset_path: str,
                         test_dataset_path: Optional[str] = None,
                         model: str = "gpt-3.5-turbo-1106") -> str:
        """Run the complete fine-tuning pipeline"""
        print("=== Starting Full Fine-tuning Pipeline ===")
        
        # Step 1: Validate dataset
        validation_results = self.validate_dataset(dataset_path)
        if validation_results["valid_examples"] == 0:
            raise ValueError("No valid examples found in dataset")
        
        # Step 2: Upload dataset
        file_id = self.upload_dataset(dataset_path)
        
        # Step 3: Start fine-tuning
        job_id = self.start_fine_tuning(file_id, model=model)
        
        # Step 4: Monitor progress
        self.monitor_fine_tuning(job_id)
        
        # Step 5: Get final model ID
        job = self.client.fine_tuning.jobs.retrieve(job_id)
        if job.status == "succeeded":
            model_id = job.fine_tuned_model
            
            # Step 6: Test with sample cases
            test_cases = [
                "قال الله تعالى: بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                "روى البخاري أن النبي قال: إنما الأعمال بالنيات",
                "هذا نص عام عن الإسلام بدون آيات أو أحاديث",
                "قال تعالى: الحمد لله رب العالمين وقال النبي: الدين المعاملة"
            ]
            self.test_model(model_id, test_cases)
            
            # Step 7: Evaluate if test dataset provided
            if test_dataset_path:
                self.evaluate_model(model_id, test_dataset_path)
            
            return model_id
        else:
            raise RuntimeError("Fine-tuning failed")


def main():
    """Main function to run the fine-tuning process"""
    
    # Configuration
    DATASET_PATH = "./output/phrase_detection_train.jsonl"
    MODEL_NAME = "gpt-4.1-mini-2025-04-14"
    
    try:
        # Initialize fine-tuner
        fine_tuner = PhraseDetectionFineTuner()
        
        # Run the pipeline
        model_id = fine_tuner.run_full_pipeline(
            dataset_path=DATASET_PATH,
            model=MODEL_NAME
        )
        
        print(f"\n=== Fine-tuning Complete ===")
        print(f"Model ID: {model_id}")
        print("You can now use this model for phrase detection!")

        # save the model ID with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"fine_tuned_model_ids.txt", "a") as f:
            f.write(json.dumps({
                "model_id": model_id,
                "timestamp": timestamp
            }, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Error in fine-tuning pipeline: {str(e)}")


if __name__ == "__main__":
    main()
