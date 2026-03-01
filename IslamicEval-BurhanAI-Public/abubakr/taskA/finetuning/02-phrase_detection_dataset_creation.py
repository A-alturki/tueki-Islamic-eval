"""
Phrase Detection Dataset Creation for Arabic Religious Text (Updated)
====================================================================

This script creates a comprehensive dataset for fine-tuning a model to detect
Quranic verses (Ayahs) and Prophetic sayings (Hadiths) in Arabic text.

The dataset includes:
1. Original competition data (Tasks A, B, C)
2. LLM-generated synthetic examples
3. Augmented edge cases and negative examples
4. Arabic text normalization variations
5. Balanced representation with/without tashkeel (diacritics)

The script generates both training (85%) and development (15%) datasets.
"""

import pandas as pd
import json
import re
import random
from typing import List, Dict, Any, Tuple
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


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
    
    @staticmethod
    def add_partial_tashkeel(text: str) -> str:
        """Add tashkeel to some random words (simulate real-world mixed text)"""
        # This is a simplified version - in practice, you'd use proper Arabic morphology
        words = text.split()
        for i in range(len(words)):
            if random.random() < 0.3:  # 30% chance to add some diacritics
                # Add some common patterns (simplified)
                if 'الله' in words[i]:
                    words[i] = words[i].replace('الله', 'اللَّهِ')
                elif len(words[i]) > 3:
                    # Add random diacritic to middle character
                    mid = len(words[i]) // 2
                    diacritics = ['َ', 'ُ', 'ِ', 'ْ']
                    words[i] = words[i][:mid] + random.choice(diacritics) + words[i][mid:]
        return ' '.join(words)


class EnhancedDatasetCreator:
    """Main class for creating the enhanced phrase detection dataset"""
    
    def __init__(self):
        self.normalizer = ArabicTextNormalizer()
        self.original_data = []
        self.llm_generated_data = []
        self.augmented_data = []
        
    def load_competition_data(self) -> None:
        """Load and process original competition datasets"""
        print("Loading competition data...")
        
        # Load unified datasets (assuming they exist from the notebook)
        try:
            df_a = pd.read_csv("../../datasets/taskA_unified.tsv", sep='\t')
            df_b = pd.read_csv("../../datasets/taskB_unified.tsv", sep='\t')
            df_c = pd.read_csv("../../datasets/taskC_unified.tsv", sep='\t')
            
            # Process each dataset
            for df, task_name in [(df_a, 'A'), (df_b, 'B'), (df_c, 'C')]:
                self._process_competition_dataset(df, task_name)
                
            print(f"Loaded {len(self.original_data)} original examples")
                
        except FileNotFoundError:
            print("Competition datasets not found. Please run the notebook first.")
            return
    
    def _process_competition_dataset(self, df: pd.DataFrame, task_name: str) -> None:
        """Process individual competition dataset"""
        # Group by Response to combine all detected phrases for same text
        response_groups = {}
        
        for _, row in df.iterrows():
            # Skip rows without annotations
            if pd.isna(row.get('Label')) or row.get('Label') == 'NoAnnotation':
                continue
                
            # Normalize labels to just 'Ayah' or 'Hadith'
            label = self._normalize_label(row['Label'])
            if not label:
                continue
            
            response_text = row['Response']
            
            # Initialize group if not exists
            if response_text not in response_groups:
                response_groups[response_text] = []
            
            # Extract span information from TSV
            span_start = row.get('Span_Start', -1)
            span_end = row.get('Span_End', -1)
            original_span = row.get('Original_Span', '')
            
            # Convert to int if they're strings
            try:
                span_start = int(span_start) if span_start != -1 else -1
                span_end = int(span_end) if span_end != -1 else -1
            except (ValueError, TypeError):
                span_start = span_end = -1
            
            # Add phrase to group with span information
            phrase = {
                "label": label,
                "value": original_span,
                "span_start": span_start,
                "span_end": span_end
            }
            
            # Avoid duplicates
            if phrase not in response_groups[response_text]:
                response_groups[response_text].append(phrase)
        
        # Create examples from grouped data with span_order
        for response_text, detected_phrases in response_groups.items():
            # Sort phrases by span_start to assign proper span_order
            detected_phrases.sort(key=lambda x: x.get('span_start', 0))
            
            # Add span_order
            for i, phrase in enumerate(detected_phrases):
                phrase['span_order'] = i
            
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Extract Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from the given text. Return only the exact quoted religious content."
                    },
                    {
                        "role": "user", 
                        "content": response_text
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "detected_phrases": detected_phrases
                        }, ensure_ascii=False)
                    }
                ],
                "source": f"original_task_{task_name}",
                "type": self._get_example_type(response_text, detected_phrases)
            }
            
            self.original_data.append(example)
    
    def _normalize_label(self, label: str) -> str:
        """Convert all label variations to just 'Ayah' or 'Hadith'"""
        if 'Ayah' in label:
            return 'Ayah'
        elif 'Hadith' in label:
            return 'Hadith'
        return None
    
    def load_llm_generated_data(self, llm_dataset_path: str = "llm_generated_dataset.jsonl") -> None:
        """Load LLM-generated synthetic examples"""
        print("Loading LLM-generated data...")
        
        try:
            with open(llm_dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line.strip())
                    
                    # Validate and clean the example
                    cleaned_example = self._validate_and_clean_llm_example(example)
                    if cleaned_example:
                        # Add metadata
                        cleaned_example["source"] = "llm_generated"
                        cleaned_example["type"] = self._get_example_type_from_messages(cleaned_example["messages"])
                        
                        self.llm_generated_data.append(cleaned_example)
            
            print(f"Loaded {len(self.llm_generated_data)} LLM-generated examples")
        
        except FileNotFoundError:
            print(f"LLM dataset not found: {llm_dataset_path}")
            print("Run phrase_detection_dataset_llm_generation.py first")
    
    def _validate_and_clean_llm_example(self, example: Dict) -> Dict:
        """Validate and clean LLM-generated example"""
        try:
            messages = example.get("messages", [])
            if len(messages) != 3:
                return None
            
            # Parse assistant content
            assistant_content = json.loads(messages[2]["content"])
            detected_phrases = assistant_content.get("detected_phrases", [])
            
            # Clean and validate detected phrases
            valid_phrases = []
            for phrase in detected_phrases:
                # Skip if phrase is just a string (invalid format)
                if isinstance(phrase, str):
                    continue
                
                # Skip if phrase doesn't have required fields
                if not isinstance(phrase, dict) or "label" not in phrase or "value" not in phrase:
                    continue
                
                # Only keep Ayah or Hadith labels
                label = phrase.get("label", "")
                if label in ["Ayah", "Hadith"]:
                    # Ensure all required span fields exist
                    cleaned_phrase = {
                        "label": label,
                        "value": phrase["value"],
                        "span_order": phrase.get("span_order", len(valid_phrases)),
                        "span_start": phrase.get("span_start", -1),
                        "span_end": phrase.get("span_end", -1)
                    }
                    valid_phrases.append(cleaned_phrase)
            
            # Create cleaned example
            cleaned_example = {
                "messages": [
                    messages[0],  # system message
                    messages[1],  # user message
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "detected_phrases": valid_phrases
                        }, ensure_ascii=False)
                    }
                ]
            }
            
            return cleaned_example
            
        except (json.JSONDecodeError, KeyError, IndexError):
            return None
    
    def _get_example_type_from_messages(self, messages: List[Dict]) -> str:
        """Determine example type from messages"""
        try:
            assistant_content = json.loads(messages[2]["content"])
            phrases = assistant_content.get("detected_phrases", [])
            return self._get_example_type(messages[1]["content"], phrases)
        except:
            return "unknown"
    
    def _get_example_type(self, text: str, phrases: List[Dict]) -> str:
        """Classify example type based on content"""
        if not phrases:
            return "negative"
        elif len(phrases) > 1:
            return "multi_phrase"
        elif phrases[0]["label"] == "Ayah":
            return "ayah_only"
        elif phrases[0]["label"] == "Hadith":
            return "hadith_only"
        else:
            return "unknown"
    
    def create_additional_augmentations(self) -> None:
        """Create additional augmented examples to balance the dataset"""
        print("Creating additional augmentations...")
        
        # Create negative examples from original data
        self._create_negative_from_original()
        
        # Create normalization variants
        self._create_normalization_variants()
        
        print(f"Created {len(self.augmented_data)} additional augmented examples")
    
    def _create_negative_from_original(self) -> None:
        """Create negative examples by modifying original texts to remove quotes"""
        negative_texts = [
            "الإسلام دين الرحمة والعدل، وقد جاء ليهدي الناس إلى الصراط المستقيم.",
            "في الفقه الإسلامي توجد مذاهب مختلفة تفسر النصوص الشرعية حسب منهجياتها.",
            "التربية الإسلامية تهدف إلى بناء شخصية متوازنة تجمع بين الدنيا والآخرة.",
            "العلماء اتفقوا على أهمية طلب العلم في الإسلام وضرورة نشره بين الناس.",
            "الحج من أركان الإسلام الخمسة ويجب على كل مسلم مستطيع أداؤه مرة في العمر."
        ]
        
        for text in negative_texts:
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Extract Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from the given text. Return only the exact quoted religious content."
                    },
                    {
                        "role": "user",
                        "content": text
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps({
                            "detected_phrases": []
                        }, ensure_ascii=False)
                    }
                ],
                "source": "augmented_negative",
                "type": "negative"
            }
            self.augmented_data.append(example)
    
    def _create_normalization_variants(self) -> None:
        """Create variants of existing examples with different normalizations"""
        sample_religious_texts = {
            "Ayah": [
                "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"
            ],
            "Hadith": [
                "إنما الأعمال بالنيات",
                "الدين المعاملة"
            ]
        }
        
        for label, texts in sample_religious_texts.items():
            for text in texts:
                contexts = [
                    f"قال الله تعالى: {text}" if label == "Ayah" else f"قال رسول الله: {text}",
                    f"ومن الأدلة على ذلك: {text}"
                ]
                
                context = random.choice(contexts)
                
                # Calculate span positions
                phrase_start = context.find(text)
                phrase_end = phrase_start + len(text)
                
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Extract Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from the given text. Return only the exact quoted religious content."
                        },
                        {
                            "role": "user",
                            "content": context
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps({
                                "detected_phrases": [
                                    {
                                        "label": label,
                                        "value": text,
                                        "span_order": 0,
                                        "span_start": phrase_start,
                                        "span_end": phrase_end
                                    }
                                ]
                            }, ensure_ascii=False)
                        }
                    ],
                    "source": "augmented_variant",
                    "type": "ayah_only" if label == "Ayah" else "hadith_only"
                }
                self.augmented_data.append(example)
    
    def combine_all_datasets(self) -> List[Dict]:
        """Combine all data sources"""
        print("Combining all datasets...")
        
        all_data = self.original_data + self.llm_generated_data + self.augmented_data
        
        # Shuffle for better distribution
        random.shuffle(all_data)
        
        print(f"Total combined examples: {len(all_data)}")
        print(f"  Original: {len(self.original_data)}")
        print(f"  LLM-generated: {len(self.llm_generated_data)}")
        print(f"  Augmented: {len(self.augmented_data)}")
        
        return all_data
    
    def split_train_dev(self, data: List[Dict], dev_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train and dev sets with balanced distribution"""
        print(f"Splitting data into train ({int((1-dev_ratio)*100)}%) and dev ({int(dev_ratio*100)}%) sets...")
        
        # Group by type for stratified split
        type_groups = {}
        for example in data:
            example_type = example.get("type", "unknown")
            if example_type not in type_groups:
                type_groups[example_type] = []
            type_groups[example_type].append(example)
        
        train_data = []
        dev_data = []
        
        # Split each type proportionally
        for example_type, examples in type_groups.items():
            if len(examples) < 2:
                # If too few examples, put all in train
                train_data.extend(examples)
                continue
            
            # Simple random split if sklearn is not available
            random.shuffle(examples)
            split_idx = int(len(examples) * (1 - dev_ratio))
            type_train = examples[:split_idx]
            type_dev = examples[split_idx:]
            
            train_data.extend(type_train)
            dev_data.extend(type_dev)
            
            print(f"  {example_type}: {len(type_train)} train, {len(type_dev)} dev")
        
        # Final shuffle
        random.shuffle(train_data)
        random.shuffle(dev_data)
        
        print(f"Final split: {len(train_data)} train, {len(dev_data)} dev")
        
        return train_data, dev_data
    
    def save_datasets(self, train_data: List[Dict], dev_data: List[Dict], 
                     train_path: str = "./output/phrase_detection_train.jsonl",
                     dev_path: str = "./output/phrase_detection_dev.jsonl") -> None:
        """Save train and dev datasets"""
        print("Saving datasets...")
        
        # Save train dataset
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train_data:
                # Remove metadata before saving
                clean_example = {
                    "messages": example["messages"]
                }
                f.write(json.dumps(clean_example, ensure_ascii=False) + '\n')
        
        # Save dev dataset
        with open(dev_path, 'w', encoding='utf-8') as f:
            for example in dev_data:
                # Remove metadata before saving
                clean_example = {
                    "messages": example["messages"]
                }
                f.write(json.dumps(clean_example, ensure_ascii=False) + '\n')
        
        print(f"Train dataset saved to: {train_path}")
        print(f"Dev dataset saved to: {dev_path}")
        
        # Save statistics
        self._save_dataset_statistics(train_data, dev_data, train_path, dev_path)
    
    def _save_dataset_statistics(self, train_data: List[Dict], dev_data: List[Dict],
                                train_path: str, dev_path: str) -> None:
        """Save comprehensive dataset statistics"""
        
        def analyze_dataset(data: List[Dict], name: str) -> Dict:
            stats = {
                "total_examples": len(data),
                "by_type": {},
                "by_source": {},
                "ayah_examples": 0,
                "hadith_examples": 0,
                "negative_examples": 0,
                "multi_phrase_examples": 0
            }
            
            for example in data:
                # Count by type
                example_type = example.get("type", "unknown")
                stats["by_type"][example_type] = stats["by_type"].get(example_type, 0) + 1
                
                # Count by source
                source = example.get("source", "unknown")
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
                
                # Count phrase types
                try:
                    assistant_content = json.loads(example["messages"][2]["content"])
                    phrases = assistant_content.get("detected_phrases", [])
                    
                    if not phrases:
                        stats["negative_examples"] += 1
                    elif len(phrases) > 1:
                        stats["multi_phrase_examples"] += 1
                    
                    for phrase in phrases:
                        if phrase.get("label") == "Ayah":
                            stats["ayah_examples"] += 1
                        elif phrase.get("label") == "Hadith":
                            stats["hadith_examples"] += 1
                
                except (json.JSONDecodeError, KeyError):
                    continue
            
            return stats
        
        train_stats = analyze_dataset(train_data, "train")
        dev_stats = analyze_dataset(dev_data, "dev")
        
        combined_stats = {
            "train": train_stats,
            "dev": dev_stats,
            "total": {
                "total_examples": len(train_data) + len(dev_data),
                "train_examples": len(train_data),
                "dev_examples": len(dev_data),
                "train_ratio": len(train_data) / (len(train_data) + len(dev_data)),
                "dev_ratio": len(dev_data) / (len(train_data) + len(dev_data))
            }
        }
        
        stats_path = train_path.replace('.jsonl', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, ensure_ascii=False, indent=2)
        
        print(f"Statistics saved to: {stats_path}")
        
        # Print summary
        print(f"\n=== Dataset Statistics ===")
        print(f"Train: {train_stats['total_examples']} examples")
        print(f"  Ayah: {train_stats['ayah_examples']}")
        print(f"  Hadith: {train_stats['hadith_examples']}")
        print(f"  Negative: {train_stats['negative_examples']}")
        print(f"  Multi-phrase: {train_stats['multi_phrase_examples']}")
        
        print(f"Dev: {dev_stats['total_examples']} examples")
        print(f"  Ayah: {dev_stats['ayah_examples']}")
        print(f"  Hadith: {dev_stats['hadith_examples']}")
        print(f"  Negative: {dev_stats['negative_examples']}")
        print(f"  Multi-phrase: {dev_stats['multi_phrase_examples']}")

    def create_complete_dataset(self, llm_dataset_path: str = "./output/llm_generated_dataset.jsonl") -> None:
        """Main method to create the complete enhanced dataset"""
        print("=== Starting Enhanced Dataset Creation ===")
        
        # Load all data sources
        self.load_competition_data()
        self.load_llm_generated_data(llm_dataset_path)
        self.create_additional_augmentations()
        
        # Combine all sources
        all_data = self.combine_all_datasets()
        
        # Split into train/dev
        train_data, dev_data = self.split_train_dev(all_data)
        
        # Save datasets
        self.save_datasets(train_data, dev_data)
        
        print("=== Enhanced Dataset Creation Complete ===")


def main():
    """Main function to run enhanced dataset creation"""
    
    # Configuration
    LLM_DATASET_PATH = "./output/llm_generated_dataset.jsonl"
    
    try:
        # Create the enhanced dataset
        creator = EnhancedDatasetCreator()
        creator.create_complete_dataset(llm_dataset_path=LLM_DATASET_PATH)
        
        print("\nEnhanced dataset creation completed!")
        print("Generated files:")
        print("  - ./output/phrase_detection_train.jsonl")
        print("  - ./output/phrase_detection_dev.jsonl")
        print("  - ./output/phrase_detection_train_stats.json")
        
    except Exception as e:
        print(f"Error in enhanced dataset creation: {str(e)}")


if __name__ == "__main__":
    main()
