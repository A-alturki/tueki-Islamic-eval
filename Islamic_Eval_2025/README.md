# Islamic Text Verification System

A professional Python application for verifying and processing Quranic verses and Hadith texts using Ollama local LLM models.

## Project Structure

```
islamic/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration and settings
│   ├── text_processor.py  # Text cleaning and processing utilities
│   ├── verse_merger.py    # Logic for merging consecutive verses
│   ├── prompts.py         # LLM prompt templates
│   └── verifier.py        # Main verification logic
├── dataset/
│   └── dev_top20_matches.pkl  # Input dataset file
├── results/
│   ├── gemma3_1b-it-fp16_with_diacritic/    # Results with diacritics removed
│   ├── gemma3_1b-it-fp16_without_diacritic/ # Results with diacritics kept
│   └── [model_name]_[diacritic_setting]/    # Auto-generated based on config
├── main.py                # Entry point script
├── requirements.txt       # Project dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Setup

1. **Install Ollama** (if not already installed):
   - Download from [https://ollama.ai](https://ollama.ai)
   - Pull the required model: `ollama pull MODEL_NAME`

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
   - Copy `.env.example` to `.env`
   - Adjust settings as needed (model, base URL, etc.)

4. **Run the application**:
```bash
python main.py
```

## Input Data

The system expects a PKL file named `dev_top20_matches.pkl` containing the queries and candidate matches to process.

## Output

Results are saved in compressed joblib format as `final.jbl` in the results folder.
