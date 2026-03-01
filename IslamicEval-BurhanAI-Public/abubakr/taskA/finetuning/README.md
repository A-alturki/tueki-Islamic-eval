# Phrase Detection Dataset Creation Process

This document explains the comprehensive process for creating a robust dataset for fine-tuning a model to detect Quranic verses (Ayahs) and Prophetic sayings (Hadiths) in Arabic text.

## Overview

The dataset creation process combines:
1. **Original competition data** from Tasks A, B, and C
2. **Augmented examples** including negative cases and edge cases
3. **Arabic text variations** with different normalizations and tashkeel handling
4. **Balanced representation** ensuring robustness across text variations

## Dataset Structure

### Input Format
Each training example follows OpenAI's fine-tuning format:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "Extract Quranic verses (Ayahs) and Prophetic sayings (Hadiths) from the given text. Return only the exact quoted religious content."
    },
    {
      "role": "user",
      "content": "Arabic text containing potential religious phrases"
    },
    {
      "role": "assistant",
      "content": "{\"detected_phrases\": [{\"label\": \"Ayah\", \"value\": \"verse text\"}]}"
    }
  ]
}
```

### Output Format
The model outputs JSON with detected phrases:
```json
{
  "detected_phrases": [
    {
      "label": "Ayah",
      "value": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    },
    {
      "label": "Hadith", 
      "value": "إنما الأعمال بالنيات"
    }
  ]
}
```

## Data Sources

### 1. Original Competition Data (70% of dataset)

**Source Files:**
- `TaskA_Input.xml` + `TaskA_GT.tsv` - Phrase identification
- `TaskB_Input.xml` + `TaskB_GT.tsv` - Accuracy validation
- `TaskC_Input.xml` + `TaskC_GT.tsv` - Error correction

**Label Normalization:**
- `Ayah` → `Ayah`
- `CorrectAyah`, `WrongAyah` → `Ayah`
- `Hadith` → `Hadith`
- `CorrectHadith`, `WrongHadith` → `Hadith`
- `NoAnnotation` → Excluded

**Processing Steps:**
1. Parse XML files to extract Question_ID, Response text
2. Merge with ground truth TSV files
3. Extract actual text spans using `Span_Start` and `Span_End`
4. Normalize labels to binary classification (Ayah/Hadith)
5. Convert to OpenAI fine-tuning format

### 2. Negative Examples (15% of dataset)

**Purpose:** Teach the model when NOT to detect religious content

**Auto-generated Examples:**
- General Islamic discussions without quotes
- Historical narratives about Islamic figures
- Scholarly opinions and fiqh discussions
- Contemporary Islamic issues
- Academic references without actual text

**Sample Texts:**
```arabic
"الإسلام دين السلام والرحمة، وقد جاء ليهدي الناس إلى طريق الحق والخير."
"يجب على المسلم أن يؤدي الصلاة في أوقاتها المحددة حسب تعاليم الدين الإسلامي."
"العلماء اختلفوا في هذه المسألة، فمنهم من قال بالجواز ومنهم من منع ذلك."
```

**Expected Output:** `{"detected_phrases": []}`

### 3. Arabic Text Variations (10% of dataset)

**Purpose:** Handle Arabic text diversity and normalization

#### 3.1 Tashkeel (Diacritics) Variations
- **With full tashkeel:** `بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ`
- **Without tashkeel:** `بسم الله الرحمن الرحيم`
- **Partial tashkeel:** `بِسم اللَّه الرحمن الرحيم`

#### 3.2 Arabic Normalization
- **Ya normalization:** `ى` → `ي`
- **Alef variations:** `أ`, `إ`, `آ` → `ا`
- **Hamza normalization:** `ؤ` → `و`, `ئ` → `ي`
- **Taa Marbouta:** `ة` → `ه` (in some contexts)

#### 3.3 Sample Religious Texts Used
**Ayahs:**
- `بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ`
- `الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ`
- `وَمَا خَلَقْتُ الْجِنَّ وَالْإِنسَ إِلَّا لِيَعْبُدُونِ`
- `إِنَّ مَعَ الْعُسْرِ يُسْرًا`

**Hadiths:**
- `إنما الأعمال بالنيات`
- `المسلم من سلم المسلمون من لسانه ويده`
- `خير الناس أنفعهم للناس`
- `الدين المعاملة`

### 4. Edge Cases (5% of dataset)

**Purpose:** Handle ambiguous and challenging cases

#### 4.1 Reference-Only Cases (No Detection Expected)
```arabic
"كما ذكر في سورة البقرة آية رقم 255"
"روى البخاري في صحيحه هذا الحديث"
"وفي هذا المعنى آية في سورة النور"
```

#### 4.2 Incomplete/Partial Quotes
```arabic
"قال تعالى: إنا أعطيناك... وذكر باقي الآية"
"من الأحاديث المشهورة: إنما الأعمال... الحديث"
```

#### 4.3 Out-of-Domain Religious Content
```
"In the beginning was the Word, and the Word was with God"
"Shema Yisrael, Adonai Eloheinu, Adonai Echad"
```

#### 4.4 Poetry and Mixed Language
```arabic
"يا من له عز الشفاعة وحده وإليه في يوم المعاد المفزع"
"The Prophet Muhammad (peace be upon him) said: الدين المعاملة"
```

## Implementation Details

### Arabic Text Normalizer Class

**Functions:**
- `remove_tashkeel()` - Removes all Arabic diacritics
- `normalize_arabic()` - Applies standard Arabic normalizations
- `add_partial_tashkeel()` - Adds random diacritics for variation

### Dataset Creator Class

**Main Methods:**
1. `load_competition_data()` - Process original XML/TSV files
2. `create_negative_examples()` - Generate non-religious content
3. `create_arabic_variations()` - Create normalized versions
4. `create_edge_cases()` - Handle ambiguous cases
5. `create_multi_phrase_examples()` - Multiple phrases per text
6. `combine_and_save_dataset()` - Final dataset assembly

## Data Balancing Strategy

### Distribution Target:
- **70%** Original competition data with religious content
- **15%** Negative examples (no religious content)
- **10%** Arabic text variations
- **5%** Edge cases and challenging examples

### Tashkeel Balance:
- **50%** Examples with tashkeel (diacritics)
- **50%** Examples without tashkeel

## Quality Assurance

### Validation Checks:
1. **JSON format validation** for all examples
2. **Message structure validation** (system/user/assistant roles)
3. **Label consistency** (only "Ayah" or "Hadith")
4. **Content statistics** tracking for balance verification

### Robustness Testing:
- Same Ayah/Hadith should be detected regardless of:
  - Presence/absence of tashkeel
  - Arabic normalization variations
  - Minor spelling differences
  - Contextual embedding

## Usage Instructions

### 1. Prepare Environment
```bash
pip install pandas openai python-dotenv
```

### 2. Set Up Configuration
Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. Prepare Competition Data
Ensure these files exist:
- `abubakr/datasets/taskA_unified.tsv`
- `abubakr/datasets/taskB_unified.tsv`
- `abubakr/datasets/taskC_unified.tsv`

### 4. Run Dataset Creation
```python
from phrase_detection_dataset_creation import DatasetCreator

creator = DatasetCreator()
creator.create_full_dataset()
```

### 5. Output Files
- `phrase_detection_dataset.jsonl` - Main training dataset
- `phrase_detection_dataset_stats.json` - Dataset statistics

## Integration with Original Dataset

### Merging Process:
1. **Load and normalize** original competition data
2. **Generate augmented** examples programmatically
3. **Apply text variations** to increase robustness
4. **Shuffle and combine** all data sources
5. **Validate and save** in OpenAI format

### Statistics Tracking:
The system automatically tracks:
- Total examples count
- Distribution by source (original vs augmented)
- Label distribution (Ayah/Hadith/Negative)
- Text variation statistics

## Expected Outcomes

### Model Robustness:
- **Language variation tolerance** - Works with/without tashkeel
- **Context awareness** - Distinguishes quotes from references
- **Precision focus** - Avoids false positives in general Islamic text
- **Multi-phrase detection** - Handles multiple Ayahs/Hadiths per text

### Performance Metrics:
- **High precision** for exact religious quote detection
- **Balanced recall** across Ayah and Hadith categories
- **Normalization invariance** across Arabic text variations
- **Negative case handling** with minimal false positives

This comprehensive approach ensures the fine-tuned model will be robust, accurate, and suitable for real-world Arabic religious text analysis.

## Model Validation and Testing Scripts

### 04-validate-finetuned-model.py

This script validates the performance of a fine-tuned OpenAI model by running predictions on a validation dataset and optionally using another model as a judge to evaluate the outputs.

#### Features:
- **Model Validation**: Tests fine-tuned models on validation datasets
- **Flexible Input**: Supports both `.jsonl` and `.tsv` input formats
- **Judge Evaluation**: Optional LLM-as-judge evaluation using GPT-4.1-mini
- **Comprehensive Output**: Generates detailed results and accuracy metrics
- **Rate Limiting**: Built-in delays to avoid API rate limits

#### Usage:
```bash
python 04-validate-finetuned-model.py <model_id> <dataset_path> [--judge]
```

#### Examples:
```bash
# Basic validation without judge
python 04-validate-finetuned-model.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK output/phrase_detection_dev.jsonl

# Validation with LLM judge evaluation
python 04-validate-finetuned-model.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK ../../datasets/taskA_unified.tsv --judge
```

#### Input Formats:
- **JSONL**: OpenAI fine-tuning format with messages array
- **TSV**: Columns: `Response`, `Label`, `Original_Span`

#### Output Files:
- `output/validation/{model_id}/llm-output.jsonl` - Detailed prediction results
- `output/validation/{model_id}/score.json` - Summary metrics and scores

#### Key Classes:
- `ModelValidator`: Main class handling validation workflow
- Methods: `load_dataset()`, `run_prediction()`, `judge_output()`, `validate_model()`

### 05-generate-subtask1A-testdata-submission.py

This script generates competition submissions for Subtask 1A (Identification of Intended Ayahs and Hadiths) using a fine-tuned model to identify character spans of religious phrases in LLM responses.

#### Features:
- **Test Data Processing**: Loads test data from JSONL format
- **Model Integration**: Uses fine-tuned models for span detection
- **Span Extraction**: Identifies exact character positions of detected phrases
- **Submission Format**: Generates TSV files following competition guidelines
- **Fallback Parsing**: Handles various model output formats with regex patterns

#### Usage:
```bash
python 05-generate-subtask1A-testdata-submission.py <model_id> [--test_file path] [--output filename] [--max-samples n] [--verbose]
```

#### Examples:
```bash
# Generate submission with default test file
python 05-generate-subtask1A-testdata-submission.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK

# Custom test file and output
python 05-generate-subtask1A-testdata-submission.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK \
  --test_file ../../datasets/taskA-testdata/Test_Subtask_1A/test_Subtask_1A.jsonl --output output/test_Subtask_1A_submission.tsv

# Process only first 10 samples for quick testing
python 05-generate-subtask1A-testdata-submission.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK \
  --test_file ../../datasets/taskA-testdata/Test_Subtask_1A/test_Subtask_1A.jsonl --output output/test_Subtask_1A_submission.tsv --max-samples 10

# Verbose mode with extracted text for validation
python 05-generate-subtask1A-testdata-submission.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK \
  --test_file ../../datasets/taskA-testdata/Test_Subtask_1A/test_Subtask_1A.jsonl --output output/test_Subtask_1A_submission.tsv --max-samples 10 --verbose
```

##### Generated Submissions:

- Tasks A:

```bash
python 05-generate-subtask1A-testdata-submission.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK   --test_file ../../datasets/taskA-testdata/Test_Subtask_1A/test_Subtask_1A.jsonl --output output/test_Subtask_1A_submission-v1.tsv --verbose
```

- Tasks B:

```bash
python 05-generate-subtask1A-testdata-submission.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK   --test_file ../../datasets/taskA-testdata/Test_Subtask_1B/Test_Subtask_1B.jsonl --output output/test_Subtask_1B_submission-v1.tsv --verbose
```

- Tasks C:

```bash
python 05-generate-subtask1A-testdata-submission.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK   --test_file ../../datasets/taskA-testdata/Test_Subtask_1C/Test_Subtask_1C.jsonl --output output/test_Subtask_1C_submission-v1.tsv --verbose
```

#### Input Format:
Test data JSONL with fields:
- `ID`: Question identifier (e.g., "A-Q001")
- `Model`: Source model name
- `Text`: Original question
- `Response`: LLM response text to analyze

#### Output Format:
TSV file with columns:
- `Question_ID`: Question identifier from input
- `Span_Start`: Zero-based start character index (inclusive)
- `Span_End`: Zero-based end character index (inclusive)
- `Span_Type`: "Ayah", "Hadith", or "No_Spans"
- `Original_Span`: (Verbose mode only) The actual extracted text for validation

#### Key Classes:
- `Subtask1ASubmissionGenerator`: Main submission generation class
- Methods: `extract_spans_with_model()`, `parse_model_output()`, `find_spans_in_text()`

#### Span Detection Logic:
1. **Model Prediction**: Uses fine-tuned model to identify religious phrases
2. **Output Parsing**: Attempts JSON parsing, falls back to regex patterns
3. **Span Location**: Finds exact character positions in original text
4. **Competition Format**: Outputs according to Subtask 1A requirements

### 06-generate-submission-html-preview.py

This script generates an interactive HTML preview for Subtask 1A submissions to help debug and review results with a clean Bootstrap interface. It provides a visual way to quickly assess model performance and identify issues.

#### Features:
- **Bootstrap UI**: Clean, responsive design with cards and tables
- **Arabic Text Support**: Proper RTL direction and Arabic-friendly fonts
- **Span Highlighting**: Color-coded badges for Ayah (green) and Hadith (blue)
- **Summary Statistics**: Overview cards showing total examples and span type counts
- **Side-by-side View**: Question and highlighted response in columns
- **Detailed Analysis**: Shows start/end positions, type, original text, and LLM values

#### Usage:
```bash
python 06-generate-submission-html-preview.py --submission-file <path> --test-file <path> [--output <filename>]
```

#### Examples:
```bash
# Generate HTML preview for submission
python 06-generate-submission-html-preview.py \
  --submission-file ./output/test_Subtask_1A_submission-v1.tsv \
  --test-file ../../datasets/taskA-testdata/Test_Subtask_1A/test_Subtask_1A.jsonl \
  --output output/submission_preview.html
```

#### Input Requirements:
- **Submission File**: TSV file generated by `05-generate-subtask1A-testdata-submission.py`
- **Test File**: Original JSONL test data with questions and responses

#### Output:
Single HTML file with:
- Summary statistics dashboard
- Color-coded legend for span types
- Individual example cards showing:
  - Original question and response
  - Highlighted spans in the text
  - Detailed spans table with positions and extracted text

#### Key Classes:
- `SubmissionHTMLPreview`: Main HTML generation class
- Methods: `highlight_spans()`, `generate_html()`, `_generate_example_html()`

Both scripts require:
- OpenAI API key (via `.env` file or environment variable)  
- Required Python packages: `openai`, `pandas`, `python-dotenv`, `json_repair`


