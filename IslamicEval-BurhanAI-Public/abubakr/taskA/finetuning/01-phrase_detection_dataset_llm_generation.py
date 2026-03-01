"""
Phrase Detection Dataset LLM Generation
=======================================

This script            messages = [
                {"role": "system", "content": "You are an Islamic scholar creating training data for Arabic religious text detection."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:5])
        
        return examples[:count]I GPT-4o-mini to generate synthetic dataset samples
for enhancing the phrase detection model training. It creates variations
and edge cases to improve model robustness.

The script generates samples using different prompting strategies to ensure
comprehensive coverage of all cases and edge cases.
"""

import json
import random
import time
import re
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import dotenv_values
import os
import json_repair


class LLMDatasetGenerator:
    """Generate synthetic dataset samples using GPT-4o-mini"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key"""
        if api_key is None:
            config = dotenv_values("../../.env")
            api_key = config.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4.1"
        self.generated_samples = []
        
        # Rate limiting
        self.request_delay = 1  # seconds between requests
        
        # Progressive saving
        self.output_file = None
        self.total_saved = 0
        
        # Line range tracking for logging
        self.line_ranges = {}
        self.current_line = 0
        
    def _make_api_call(self, messages: List[Dict[str, str]]):
        """Make API call with error handling and rate limiting"""
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
            )
            
            # Rate limiting
            time.sleep(self.request_delay)

            llm_response = response.choices[0].message.content
            json_repaired = json_repair.loads(llm_response)

            if json_repaired:
                return json_repaired

            return None
        
        except Exception as e:
            print(f"API call error: {str(e)}")
            return None
    
    def generate_positive_examples_with_ayahs(self, count: int = 50) -> List[Dict]:
        """Generate examples containing Quranic verses"""
        print(f"Generating {count} positive examples with Ayahs...")
        
        prompt = """Generate realistic Arabic text examples that contain authentic Quranic verses (Ayahs).

Requirements:
1. Each example should include 1-2 actual Quranic verses embedded naturally in Arabic text
2. Use well-known Ayahs like Al-Fatiha, Ayat al-Kursi, or verses from Al-Baqarah
3. Embed them in contexts like sermons, Islamic discussions, or religious explanations
4. Vary the context: some formal, some informal, some scholarly
5. Include both cases with and without tashkeel (diacritics)
6. Use natural Arabic introducing phrases like "قال الله تعالى", "وقد ذكر في القرآن", etc.

Output format:
{
  "examples": [
    {
      "text": "Arabic text with embedded Ayah",
      "detected_phrases": [
        {
          "label": "Ayah",
          "value": "exact Quranic verse text"
        }
      ]
    }
  ]
}

Generate 5 diverse examples."""

        examples = []
        iterations = (count + 4) // 5  # Generate 5 at a time
        
        for i in range(iterations):
            print(f"  Generating batch {i+1}/{iterations}...")
            
            messages = [
                {"role": "system", "content": "You are an Islamic scholar creating training data for Arabic religious text detection."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:5])  # Take max 5
        
        return examples[:count]
    
    def generate_positive_examples_with_hadiths(self, count: int = 50) -> List[Dict]:
        """Generate examples containing authentic Hadiths"""
        print(f"Generating {count} positive examples with Hadiths...")
        
        prompt = """Generate realistic Arabic text examples that contain authentic Prophetic sayings (Hadiths).

Requirements:
1. Each example should include 1-2 actual Hadiths embedded naturally in Arabic text
2. Use well-known Hadiths like "إنما الأعمال بالنيات", "المسلم من سلم المسلمون من لسانه ويده", etc.
3. Embed them in contexts like Islamic lectures, fatwa discussions, or religious advice
4. Vary the context: some quoting Bukhari/Muslim, some general religious guidance
5. Include both cases with and without tashkeel (diacritics)
6. Use natural Arabic introducing phrases like "قال رسول الله", "روى البخاري", "في الحديث الشريف", etc.

Output format:
{
  "examples": [
    {
      "text": "Arabic text with embedded Hadith",
      "detected_phrases": [
        {
          "label": "Hadith",
          "value": "exact Hadith text"
        }
      ]
    }
  ]
}

Generate 5 diverse examples."""

        examples = []
        iterations = (count + 4) // 5
        
        for i in range(iterations):
            print(f"  Generating batch {i+1}/{iterations}...")
            
            messages = [
                {"role": "system", "content": "You are an Islamic scholar creating training data for Arabic religious text detection."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:5])
        
        return examples[:count]
    
    def generate_mixed_examples(self, count: int = 30) -> List[Dict]:
        """Generate examples with both Ayahs and Hadiths"""
        print(f"Generating {count} mixed examples with both Ayahs and Hadiths...")
        
        prompt = """Generate realistic Arabic text examples that contain both Quranic verses AND Prophetic sayings.

Requirements:
1. Each example should include at least one Ayah and one Hadith
2. Use authentic, well-known religious texts
3. Create natural flowing text that combines both types
4. Contexts: religious lectures, Islamic education, scholarly discussions
5. Include proper attribution phrases for both types
6. Vary diacritics usage across examples

Output format:
{
  "examples": [
    {
      "text": "Arabic text with both Ayah and Hadith",
      "detected_phrases": [
        {
          "label": "Ayah",
          "value": "exact Quranic verse"
        },
        {
          "label": "Hadith",
          "value": "exact Hadith text"
        }
      ]
    }
  ]
}

Generate 3 diverse examples."""

        examples = []
        iterations = (count + 2) // 3
        
        for i in range(iterations):
            print(f"  Generating batch {i+1}/{iterations}...")
            
            messages = [
                {"role": "system", "content": "You are an Islamic scholar creating training data for Arabic religious text detection."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:3])
        
        return examples[:count]
    
    def generate_malformed_ayah_examples(self, count: int = 50) -> List[Dict]:
        """Generate examples containing malformed/incorrect Quranic verses"""
        print(f"Generating {count} malformed Ayah examples...")
        
        prompt = """Generate realistic Arabic text examples that contain INTENTIONALLY INCORRECT or MALFORMED Quranic verses (Ayahs).

Requirements:
1. Create Arabic text with INCORRECT Ayahs that appear to be Quranic but contain errors:
   - Real Ayahs with 1-3 words changed/replaced/missing
   - Real Ayahs with incorrect word order
   - Fabricated verses that sound Quranic but don't exist
   - Mix of real and fabricated parts
2. Embed them naturally in Islamic discussions, sermons, or religious explanations
3. Use authentic Quranic introducing phrases like "قال الله تعالى", "في القرآن الكريم"
4. Include references to non-existent suras or verse numbers
5. Vary between subtle errors and obvious mistakes
6. Include both cases with and without tashkeel (diacritics)

Output format:
{
  "examples": [
    {
      "text": "Arabic text with embedded INCORRECT Ayah",
      "detected_phrases": [
        {
          "label": "Ayah",
          "value": "malformed/incorrect Quranic verse text"
        }
      ]
    }
  ]
}

Generate 5 diverse examples with intentionally incorrect Ayahs."""

        examples = []
        iterations = (count + 4) // 5
        
        for i in range(iterations):
            print(f"  Generating batch {i+1}/{iterations}...")
            
            messages = [
                {"role": "system", "content": "You are creating training data for detecting incorrect religious quotes. Generate text with intentionally wrong Quranic verses for hallucination detection."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:5])
        
        return examples[:count]
    
    def generate_malformed_hadith_examples(self, count: int = 50) -> List[Dict]:
        """Generate examples containing malformed/incorrect Hadiths"""
        print(f"Generating {count} malformed Hadith examples...")
        
        prompt = """Generate realistic Arabic text examples that contain INTENTIONALLY INCORRECT or MALFORMED Prophetic sayings (Hadiths).

Requirements:
1. Create Arabic text with INCORRECT Hadiths that appear authentic but contain errors:
   - Real Hadiths with 1-3 words changed/replaced/missing
   - Real Hadiths with incorrect attribution (wrong narrator)
   - Fabricated Hadiths that sound authentic but don't exist
   - Mix of real and fabricated parts
2. Embed them naturally in Islamic lectures, fatwa discussions, or religious advice
3. Use authentic Hadith introducing phrases like "قال رسول الله", "روى البخاري", "في الحديث"
4. Include references to non-existent hadith collections or wrong narrators
5. Vary between subtle errors and obvious mistakes
6. Include both cases with and without tashkeel (diacritics)

Output format:
{
  "examples": [
    {
      "text": "Arabic text with embedded INCORRECT Hadith",
      "detected_phrases": [
        {
          "label": "Hadith",
          "value": "malformed/incorrect Hadith text"
        }
      ]
    }
  ]
}

Generate 5 diverse examples with intentionally incorrect Hadiths."""

        examples = []
        iterations = (count + 4) // 5
        
        for i in range(iterations):
            print(f"  Generating batch {i+1}/{iterations}...")
            
            messages = [
                {"role": "system", "content": "You are creating training data for detecting incorrect religious quotes. Generate text with intentionally wrong Hadiths for hallucination detection."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:5])
        
        return examples[:count]
    
    def generate_negative_examples(self, count: int = 60) -> List[Dict]:
        """Generate examples with NO religious quotes (negative cases)"""
        print(f"Generating {count} negative examples...")
        
        prompt = """Generate realistic Arabic text examples about Islamic topics that contain NO direct Quranic verses or Hadith quotes.

Requirements:
1. Create Islamic discussions without actual quotes
2. Include topics: fiqh rulings, Islamic history, religious advice, scholarly opinions
3. May reference verses/hadiths but don't quote them directly
4. Use phrases like "كما ذكر في القرآن", "روى الإمام", "في الحديث" without actual text
5. Include contemporary Islamic issues, halal/haram discussions
6. Vary length and complexity

Output format:
{
  "examples": [
    {
      "text": "Arabic Islamic discussion without quotes",
      "detected_phrases": []
    }
  ]
}

Generate 6 diverse examples."""

        examples = []
        iterations = (count + 5) // 6
        
        for i in range(iterations):
            print(f"  Generating batch {i+1}/{iterations}...")
            
            messages = [
                {"role": "system", "content": "You are an Islamic scholar creating training data for Arabic religious text detection."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:6])
        
        return examples[:count]
    
    def generate_edge_cases(self, count: int = 40) -> List[Dict]:
        """Generate challenging edge cases"""
        print(f"Generating {count} edge case examples...")
        
        prompt = """Generate challenging edge case examples for Arabic religious text detection.

Requirements:
1. Include partial/incomplete quotes with "..." 
2. References to verses/hadiths without actual text
3. Poetry that sounds religious but isn't Quran/Hadith
4. Mixed Arabic-English text
5. Very short phrases that might be ambiguous
6. Text with religious words but no actual quotes
7. Scholarly citations and footnotes
8. Modern Islamic terminology discussions

Output format:
{
  "examples": [
    {
      "text": "Challenging edge case text",
      "detected_phrases": [] // or actual phrases if they contain real quotes
    }
  ]
}

Generate 5 diverse edge cases."""

        examples = []
        iterations = (count + 4) // 5
        
        for i in range(iterations):
            print(f"  Generating batch {i+1}/{iterations}...")
            
            messages = [
                {"role": "system", "content": "You are an expert in Arabic religious text creating challenging test cases."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:5])
        
        return examples[:count]
    
    def generate_normalization_variants(self, count: int = 50) -> List[Dict]:
        """Generate variants with different Arabic normalizations"""
        print(f"Generating {count} normalization variant examples...")
        
        prompt = """Generate Arabic religious text examples with deliberate variations in spelling and diacritics.

Requirements:
1. Use the same Ayahs/Hadiths but with different Arabic text variations
2. Include: with/without tashkeel, different Ya forms (ي/ى), Alef variations (أ/إ/آ/ا)
3. Include common spelling variations that occur in real text
4. Mix formal (Classical Arabic) and informal writing styles
5. Create variants of the same religious content with different normalizations

Output format:
{
  "examples": [
    {
      "text": "Arabic text with normalization variants",
      "detected_phrases": [
        {
          "label": "Ayah/Hadith",
          "value": "religious text with specific normalization"
        }
      ]
    }
  ]
}

Generate 5 diverse examples."""

        examples = []
        iterations = (count + 4) // 5
        
        for i in range(iterations):
            print(f"  Generating batch {i+1}/{iterations}...")
            
            messages = [
                {"role": "system", "content": "You are an Arabic linguist creating text variations for ML training."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:5])
        
        return examples[:count]
    
    def generate_long_dialect_examples(self, count: int = 60) -> List[Dict]:
        """Generate longer examples (500+ words) with Arabic dialects"""
        print(f"Generating {count} long dialect examples...")
        
        dialects = [
            {
                "name": "Modern Standard Arabic",
                "description": "Contemporary formal Arabic used in media, literature, and formal discourse",
                "features": "Use formal vocabulary, complex sentence structures, and proper grammar"
            },
            {
                "name": "Egyptian dialect", 
                "description": "Egyptian colloquial Arabic with distinctive vocabulary and pronunciation",
                "features": "Use Egyptian expressions like 'إزيك', 'كده', 'عايز', and Egyptian-specific religious phrases"
            },
            {
                "name": "Gulf dialect",
                "description": "Gulf Arabic dialect from UAE, Saudi, Kuwait regions", 
                "features": "Use Gulf expressions like 'شلونك', 'وايد', 'أبي', and Gulf-specific terms"
            }
        ]
        
        examples = []
        dialect_counts = {dialect["name"]: 0 for dialect in dialects}
        target_per_dialect = count // len(dialects)
        
        iterations = (count + 2) // 3  # Generate 3 at a time
        
        for i in range(iterations):
            print(f"  Generating batch {i+1}/{iterations}...")
            
            # Select dialect with equal distribution
            available_dialects = [d for d in dialects if dialect_counts[d["name"]] < target_per_dialect]
            if not available_dialects:
                available_dialects = dialects
            
            selected_dialect = random.choice(available_dialects)
            dialect_counts[selected_dialect["name"]] += 1
            
            prompt = f"""Generate long-form Arabic text examples (minimum 500 words each) in {selected_dialect["name"]} that contain authentic Quranic verses and/or Prophetic sayings.

Dialect Requirements for {selected_dialect["name"]}:
- {selected_dialect["description"]}
- {selected_dialect["features"]}

Content Requirements:
1. Create realistic long-form content (500-800 words) such as:
   - Religious lectures or sermons
   - Islamic educational articles
   - Religious discussion forums
   - Islamic scholarship essays
   - Community religious guidance

2. Embed 2-4 authentic Quranic verses (Ayahs) and/or Prophetic sayings (Hadiths) naturally
3. Use well-known religious texts from Al-Fatiha, Al-Baqarah, authentic Hadith collections
4. Include proper context and explanation around the religious quotes
5. Mix formal religious content with the specified dialect characteristics
6. Include contemporary Islamic topics relevant to modern Muslims

Structure Examples:
- Introduction to topic → Quranic support → Hadith reinforcement → Practical application
- Discussion of Islamic principle → Multiple supporting verses → Scholar commentary → Conclusion
- Response to religious question → Detailed explanation with quotes → Practical guidance

Output format:
{{
  "examples": [
    {{
      "text": "Long Arabic text (500+ words) in {selected_dialect["name"]} with embedded religious quotes",
      "detected_phrases": [
        {{
          "label": "Ayah",
          "value": "exact Quranic verse text"
        }},
        {{
          "label": "Hadith", 
          "value": "exact Hadith text"
        }}
      ]
    }}
  ]
}}

Generate 3 diverse examples in {selected_dialect["name"]}."""

            messages = [
                {"role": "system", "content": f"You are an Islamic scholar fluent in {selected_dialect['name']} creating comprehensive religious educational content."},
                {"role": "user", "content": prompt}
            ]
            
            response = self._make_api_call(messages)
            if response:
                batch_examples = response.get("examples", []) if response else []
                examples.extend(batch_examples[:3])
        
        print(f"Dialect distribution: {dialect_counts}")
        return examples[:count]
    
    def generate_text_corruption_variants(self, base_examples: List[Dict], count: int = 80) -> List[Dict]:
        """Generate text corruption variants using text processing"""
        print(f"Generating {count} text corruption variants...")
        
        if not base_examples:
            return []
        
        corruption_variants = []
        
        for _ in range(count):
            # Select random base example
            base_example = random.choice(base_examples)
            if not base_example.get("text") or not base_example.get("detected_phrases"):
                continue
            
            corrupted_text = base_example["text"]
            corrupted_phrases = []
            
            # Apply random corruptions to detected phrases
            for span_order, phrase in enumerate(base_example["detected_phrases"]):
                original_phrase = phrase["value"]
                corrupted_phrase = self._apply_text_corruptions(original_phrase)
                
                # Update text with corrupted phrase
                corrupted_text = corrupted_text.replace(original_phrase, corrupted_phrase)
                
                corrupted_phrases.append({
                    "label": phrase["label"],
                    "value": corrupted_phrase,
                    "span_order": span_order
                })
            
            # Apply text-level corruptions
            corrupted_text = self._apply_text_level_corruptions(corrupted_text)
            
            # Calculate spans after all corruptions
            final_phrases = self._calculate_spans(corrupted_text, corrupted_phrases)
            
            corruption_variants.append({
                "text": corrupted_text,
                "detected_phrases": final_phrases
            })
        
        return corruption_variants[:count]
    
    def _apply_text_corruptions(self, text: str) -> str:
        """Apply various text corruptions to a phrase"""
        corrupted = text
        
        # Randomly apply 1-3 corruptions
        num_corruptions = random.randint(1, 3)
        corruption_types = random.sample([
            'multiple_quotes', 'unclosed_punct', 'extra_spaces', 
            'word_repetition', 'mixed_quotes', 'nested_punct',
            'char_repetition', 'arabic_punct'
        ], min(num_corruptions, 8))
        
        for corruption in corruption_types:
            if corruption == 'multiple_quotes':
                corrupted = f'"""{corrupted}""' if random.choice([True, False]) else f'"{corrupted}""'
            
            elif corruption == 'unclosed_punct':
                punct = random.choice(['"', "'", '(', '[', '{', '«'])
                if random.choice([True, False]):
                    corrupted = f'{punct}{corrupted}'  # Start only
                else:
                    corrupted = f'{corrupted}{punct}'  # End only
            
            elif corruption == 'extra_spaces':
                words = corrupted.split()
                spaced_words = []
                for word in words:
                    spaced_words.append(word)
                    if random.random() < 0.3:  # 30% chance
                        spaced_words.append(' ' * random.randint(2, 5))
                corrupted = ' '.join(spaced_words)
            
            elif corruption == 'word_repetition':
                words = corrupted.split()
                if words:
                    word_idx = random.randint(0, len(words) - 1)
                    repeat_count = random.randint(2, 4)
                    words[word_idx] = f"{words[word_idx]} " * repeat_count
                    corrupted = ' '.join(words)
            
            elif corruption == 'mixed_quotes':
                corrupted = f'"{corrupted}\'' if random.choice([True, False]) else f'\'{corrupted}"'
            
            elif corruption == 'nested_punct':
                punct_pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('«', '»')]
                start, end = random.choice(punct_pairs)
                corrupted = f'"{start}{corrupted}{end}"'
            
            elif corruption == 'char_repetition':
                # Repeat random character in a word
                words = corrupted.split()
                if words:
                    word_idx = random.randint(0, len(words) - 1)
                    word = words[word_idx]
                    if len(word) > 2:
                        char_idx = random.randint(1, len(word) - 2)
                        char = word[char_idx]
                        repeat_count = random.randint(2, 4)
                        new_word = word[:char_idx] + char * repeat_count + word[char_idx + 1:]
                        words[word_idx] = new_word
                        corrupted = ' '.join(words)
            
            elif corruption == 'arabic_punct':
                corrupted = f'«{corrupted}»' if random.choice([True, False]) else f'؟{corrupted}؟'
        
        return corrupted
    
    def _apply_text_level_corruptions(self, text: str) -> str:
        """Apply corruptions to the entire text"""
        corrupted = text
        
        # Leading/trailing spaces
        if random.random() < 0.3:
            spaces_start = ' ' * random.randint(1, 4)
            spaces_end = ' ' * random.randint(1, 4)
            corrupted = f'{spaces_start}{corrupted}{spaces_end}'
        
        # Random extra spaces between sentences
        if random.random() < 0.4:
            sentences = re.split(r'([.!?؟])', corrupted)
            for i in range(1, len(sentences), 2):
                if i + 1 < len(sentences):
                    sentences[i + 1] = ' ' * random.randint(2, 6) + sentences[i + 1].lstrip()
            corrupted = ''.join(sentences)
        
        return corrupted
    
    def _init_progressive_save(self, output_path: str) -> None:
        """Initialize progressive saving to file"""
        self.output_file = open(output_path, 'w', encoding='utf-8')
        self.total_saved = 0
        self.current_line = 0
        self.line_ranges = {}
        print(f"Initialized progressive saving to: {output_path}")
    
    def _save_batch_progressively(self, examples: List[Dict], sample_type: str = "Unknown") -> None:
        """Save a batch of examples progressively with line range tracking"""
        if not self.output_file or not examples:
            return
        
        try:
            # Convert batch to OpenAI format
            formatted_examples = self.convert_to_openai_format(examples)
            
            # Track starting line for this batch
            start_line = self.current_line + 1
            
            # Save each example immediately
            for example in formatted_examples:
                if isinstance(example, dict) and "messages" in example:
                    clean_example = {"messages": example["messages"]}
                    self.output_file.write(json.dumps(clean_example, ensure_ascii=False) + '\n')
                    self.output_file.flush()  # Ensure immediate write
                    self.total_saved += 1
                    self.current_line += 1
            
            # Track ending line for this batch
            end_line = self.current_line
            
            # Store line range for this sample type
            if sample_type not in self.line_ranges:
                self.line_ranges[sample_type] = []
            
            if formatted_examples:  # Only log if we actually saved examples
                self.line_ranges[sample_type].append({
                    "start_line": start_line,
                    "end_line": end_line,
                    "count": len(formatted_examples)
                })
                
                print(f"  Saved {len(formatted_examples)} {sample_type} examples (Lines {start_line}-{end_line}) | Total: {self.total_saved}")
            
        except Exception as e:
            print(f"Error in progressive saving: {str(e)}")
            print(f"Examples type: {type(examples)}")
            if examples:
                print(f"First example type: {type(examples[0])}")
                print(f"First example keys: {examples[0].keys() if isinstance(examples[0], dict) else 'Not a dict'}")
    
    def _close_progressive_save(self) -> None:
        """Close progressive saving file and generate summary log"""
        if self.output_file:
            self.output_file.close()
            print(f"Progressive saving completed. Total examples saved: {self.total_saved}")
            
            # Generate detailed line range summary
            print("\n" + "="*60)
            print("DATASET GENERATION SUMMARY")
            print("="*60)
            
            for sample_type, ranges in self.line_ranges.items():
                total_count = sum(r["count"] for r in ranges)
                print(f"\n{sample_type.upper()} SAMPLES:")
                print(f"  Total count: {total_count}")
                
                for i, range_info in enumerate(ranges, 1):
                    if len(ranges) > 1:
                        print(f"  Batch {i}: Lines {range_info['start_line']}-{range_info['end_line']} ({range_info['count']} samples)")
                    else:
                        print(f"  Lines {range_info['start_line']}-{range_info['end_line']} ({range_info['count']} samples)")
            
            print(f"\nTOTAL DATASET: {self.total_saved} samples")
            print("="*60)
    
    def _save_line_ranges_log(self, output_path: str) -> None:
        """Save line ranges summary to a separate log file"""
        log_path = output_path.replace('.jsonl', '_line_ranges.log')
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("DATASET GENERATION LINE RANGES SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for sample_type, ranges in self.line_ranges.items():
                total_count = sum(r["count"] for r in ranges)
                f.write(f"{sample_type.upper()} SAMPLES:\n")
                f.write(f"  Total count: {total_count}\n")
                
                for i, range_info in enumerate(ranges, 1):
                    if len(ranges) > 1:
                        f.write(f"  Batch {i}: Lines {range_info['start_line']}-{range_info['end_line']} ({range_info['count']} samples)\n")
                    else:
                        f.write(f"  Lines {range_info['start_line']}-{range_info['end_line']} ({range_info['count']} samples)\n")
                f.write("\n")
            
            f.write(f"TOTAL DATASET: {self.total_saved} samples\n")
            f.write("="*60 + "\n")
        
        print(f"Line ranges log saved to: {log_path}")
    
    def _calculate_spans(self, text: str, phrases: List[Dict]) -> List[Dict]:
        """Calculate character spans for detected phrases"""
        final_phrases = []
        
        for phrase in phrases:
            phrase_text = phrase["value"]
            start_idx = text.find(phrase_text)
            
            if start_idx != -1:
                end_idx = start_idx + len(phrase_text)
                final_phrases.append({
                    "label": phrase["label"],
                    "value": phrase_text,
                    "span_order": phrase["span_order"],
                    "span_start": start_idx,
                    "span_end": end_idx
                })
        
        return final_phrases
    
    def convert_to_openai_format(self, examples: List[Dict]) -> List[Dict]:
        """Convert generated examples to OpenAI fine-tuning format"""
        formatted_examples = []
        
        system_prompt = "Extract Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from the given text. Return only the exact quoted religious content."
        
        for example in examples:
            try:
                # Validate example structure
                if not isinstance(example, dict):
                    print(f"Skipping non-dict example: {type(example)}")
                    continue
                    
                if not example.get("text"):
                    print(f"Skipping example without text: {example}")
                    continue
                
                # Calculate spans for phrases that don't have them yet
                phrases = example.get("detected_phrases", [])
                if phrases and not all("span_start" in p for p in phrases if isinstance(p, dict)):
                    phrases = self._calculate_spans_for_example(example["text"], phrases)
                
                openai_example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": example["text"]
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps({
                                "detected_phrases": phrases
                            }, ensure_ascii=False)
                        }
                    ]
                }
                
                formatted_examples.append(openai_example)
                
            except Exception as e:
                print(f"Error processing example: {str(e)}")
                print(f"Example: {example}")
                continue
        
        return formatted_examples
    
    def _calculate_spans_for_example(self, text: str, phrases: List[Dict]) -> List[Dict]:
        """Calculate spans for phrases in original LLM examples"""
        updated_phrases = []
        
        for span_order, phrase in enumerate(phrases):
            try:
                # Validate phrase structure
                if not isinstance(phrase, dict):
                    print(f"Skipping non-dict phrase: {type(phrase)}")
                    continue
                    
                phrase_text = phrase.get("value", "")
                if not phrase_text:
                    print(f"Skipping phrase without value: {phrase}")
                    continue
                
                start_idx = text.find(phrase_text)
                
                if start_idx != -1:
                    end_idx = start_idx + len(phrase_text)
                    updated_phrases.append({
                        "label": phrase.get("label", "Unknown"),
                        "value": phrase_text,
                        "span_order": span_order,
                        "span_start": start_idx,
                        "span_end": end_idx
                    })
                else:
                    # Fallback if exact match not found
                    updated_phrases.append({
                        "label": phrase.get("label", "Unknown"),
                        "value": phrase_text,
                        "span_order": span_order,
                        "span_start": -1,
                        "span_end": -1
                    })
                    
            except Exception as e:
                print(f"Error processing phrase: {str(e)}")
                print(f"Phrase: {phrase}")
                continue
        
        return updated_phrases
    
    def generate_complete_dataset(self, target_size: int = 400, output_path: str = None) -> None:
        """Generate complete synthetic dataset with progressive saving"""
        print(f"Starting LLM dataset generation for {target_size} samples...")
        
        # Initialize progressive saving
        if output_path:
            self._init_progressive_save(output_path)
        
        # Calculate distribution (malformed = 30%, corruption variants = 20%, long dialects = 15%, rest = 35%)
        malformed_count = int(target_size * 0.30)      # 30% malformed samples
        corruption_count = int(target_size * 0.20)     # 20%
        long_dialect_count = int(target_size * 0.15)   # 15%
        remaining_size = target_size - malformed_count - corruption_count - long_dialect_count
        
        ayah_count = int(remaining_size * 0.35)        # ~12% of total
        hadith_count = int(remaining_size * 0.35)      # ~12% of total  
        mixed_count = int(remaining_size * 0.20)       # ~7% of total
        negative_count = int(remaining_size * 0.10)    # ~4% of total
        
        # Split malformed samples between Ayahs and Hadiths
        malformed_ayah_count = int(malformed_count * 0.6)    # 18% of total
        malformed_hadith_count = malformed_count - malformed_ayah_count  # 12% of total
        
        print(f"Distribution: Ayah({ayah_count}) + Hadith({hadith_count}) + Mixed({mixed_count}) + Negative({negative_count}) + MalformedAyah({malformed_ayah_count}) + MalformedHadith({malformed_hadith_count}) + LongDialect({long_dialect_count}) + Corruption({corruption_count})")
        
        # Generate and save progressively
        print("Generating Ayah examples...")
        base_ayah_examples = self.generate_positive_examples_with_ayahs(ayah_count)
        self._save_batch_progressively(base_ayah_examples, "Authentic_Ayah")
        
        print("Generating Hadith examples...")
        base_hadith_examples = self.generate_positive_examples_with_hadiths(hadith_count)
        self._save_batch_progressively(base_hadith_examples, "Authentic_Hadith")
        
        print("Generating Mixed examples...")
        base_mixed_examples = self.generate_mixed_examples(mixed_count)
        self._save_batch_progressively(base_mixed_examples, "Mixed_Ayah_Hadith")
        
        print("Generating Negative examples...")
        negative_examples = self.generate_negative_examples(negative_count)
        self._save_batch_progressively(negative_examples, "Negative_No_Quotes")
        
        print("Generating Malformed Ayah examples...")
        malformed_ayah_examples = self.generate_malformed_ayah_examples(malformed_ayah_count)
        self._save_batch_progressively(malformed_ayah_examples, "Malformed_Ayah")
        
        print("Generating Malformed Hadith examples...")
        malformed_hadith_examples = self.generate_malformed_hadith_examples(malformed_hadith_count)
        self._save_batch_progressively(malformed_hadith_examples, "Malformed_Hadith")
        
        print("Generating Long Dialect examples...")
        long_dialect_examples = self.generate_long_dialect_examples(long_dialect_count)
        self._save_batch_progressively(long_dialect_examples, "Long_Dialect")
        
        print("Generating Corruption variants...")
        positive_examples = base_ayah_examples + base_hadith_examples + base_mixed_examples + long_dialect_examples
        corruption_examples = self.generate_text_corruption_variants(positive_examples, corruption_count)
        self._save_batch_progressively(corruption_examples, "Corruption_Variants")
        
        # Close progressive saving
        self._close_progressive_save()
        
        print(f"Dataset generation completed with progressive saving!")
        
        # Keep samples for statistics (optional)
        all_examples = (base_ayah_examples + base_hadith_examples + base_mixed_examples + 
                       negative_examples + malformed_ayah_examples + malformed_hadith_examples + 
                       long_dialect_examples + corruption_examples)
        formatted_examples = self.convert_to_openai_format(all_examples)
        self.generated_samples = formatted_examples
    
    def save_dataset(self, output_path: str = "llm_generated_dataset.jsonl") -> None:
        """Save generated dataset to JSONL file"""
        print(f"Saving dataset to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in self.generated_samples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(self.generated_samples)} examples to {output_path}")
        
        # Save statistics
        stats = self._calculate_statistics()
        stats_path = output_path.replace('.jsonl', '_stats.json')
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Statistics saved to {stats_path}")
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate dataset statistics"""
        stats = {
            "total_examples": len(self.generated_samples),
            "ayah_examples": 0,
            "hadith_examples": 0,
            "negative_examples": 0,
            "multi_phrase_examples": 0,
            "corruption_examples": 0,
            "long_examples": 0,
            "dialect_examples": 0
        }
        
        for example in self.generated_samples:
            try:
                assistant_content = json.loads(example["messages"][2]["content"])
                phrases = assistant_content.get("detected_phrases", [])
                
                # Check for corruption indicators
                text = example["messages"][1]["content"]
                has_corruption = any([
                    '"""' in text, '""\'"' in text, 
                    text.count('"') % 2 != 0,
                    '  ' in text.replace('  ', ' '),
                    any(word in text for word in text.split() if text.split().count(word) > 1)
                ])
                
                if has_corruption:
                    stats["corruption_examples"] += 1
                
                # Check for long examples (500+ words)
                word_count = len(text.split())
                if word_count >= 500:
                    stats["long_examples"] += 1
                
                # Check for dialect indicators
                dialect_indicators = ['إزيك', 'كده', 'عايز', 'شلونك', 'وايد', 'أبي']
                if any(indicator in text for indicator in dialect_indicators):
                    stats["dialect_examples"] += 1
                
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

    def run_generation(self, target_size: int = 400, output_path: str = None) -> None:
        """Run complete generation pipeline with progressive saving"""
        print("=== LLM Dataset Generation Pipeline ===")
        
        if not output_path:
            output_path = "./output/llm_generated_dataset.jsonl"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.generate_complete_dataset(target_size, output_path)
        
        # Save line ranges log
        if output_path:
            self._save_line_ranges_log(output_path)
        
        # Save final statistics
        if self.generated_samples:
            stats = self._calculate_statistics()
            stats_path = output_path.replace('.jsonl', '_stats.json')
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"Statistics saved to: {stats_path}")
        
        print("=== Generation Complete ===")


def main():
    """Main function to run LLM dataset generation"""
    
    # Configuration
    TARGET_SIZE = 400  # Adjust based on your needs
    OUTPUT_PATH = "./output/llm_generated_dataset.jsonl"
    
    try:
        # Initialize generator
        generator = LLMDatasetGenerator()
        
        # Run generation
        generator.run_generation(target_size=TARGET_SIZE, output_path=OUTPUT_PATH)
        
        print(f"\nLLM dataset generation completed!")
        print(f"Generated dataset: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error in LLM generation: {str(e)}")


if __name__ == "__main__":
    main()
