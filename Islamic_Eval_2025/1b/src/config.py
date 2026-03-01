"""
Configuration module for the Islamic Text Verification System.
Handles environment variables and application settings.
"""

import os
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class Config:
    """Application configuration class."""

    # Model Selection
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")  # "ollama" or "openai"

    # Ollama Configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b-it-fp16")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # OpenAI Configuration
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # "gpt-4o" or "gpt-4o-mini"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))

    # File Paths
    INPUT_FILE = os.getenv("INPUT_FILE", "dataset/dev_top20_matches.pkl")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "final.jbl")

    # Text Processing Options
    REMOVE_DIACRITICS = os.getenv("REMOVE_DIACRITICS", "true").lower() == "true"

    @classmethod
    def get_model_name(cls, provider=None, model=None):
        """Get the current model name based on provider."""
        if provider is None:
            provider = cls.MODEL_PROVIDER

        if provider == "openai":
            return model or cls.OPENAI_MODEL
        else:  # ollama
            return model or cls.OLLAMA_MODEL

    @classmethod
    def get_results_folder(cls, provider=None, model=None):
        """Generate results folder path based on model and diacritic settings."""
        # Get model name
        model_name = cls.get_model_name(provider, model)

        # Clean model name for folder (replace : with _ and remove special chars)
        model_name = model_name.replace(":", "_").replace("/", "_").replace("-", "_")

        # Add provider prefix
        current_provider = provider or cls.MODEL_PROVIDER
        model_folder = f"{current_provider}_{model_name}"

        diacritic_suffix = "without_diacritic" if cls.REMOVE_DIACRITICS else "with_diacritic"
        folder_name = f"{model_folder}_{diacritic_suffix}"

        results_path = os.path.join("results", folder_name)

        # Create the folder if it doesn't exist
        os.makedirs(results_path, exist_ok=True)

        return results_path

    @classmethod
    def get_output_file_path(cls, filename=None, provider=None, model=None):
        """Get full output file path in the appropriate results subfolder."""
        if filename is None:
            filename = cls.OUTPUT_FILE

        results_folder = cls.get_results_folder(provider, model)
        return os.path.join(results_folder, filename)

    @classmethod
    def get_llm(cls, provider=None, model=None):
        """
        Create and return configured LLM instance.

        Args:
            provider (str, optional): Model provider ("ollama" or "openai").
                                    Defaults to MODEL_PROVIDER from config.
            model (str, optional): Specific model name. Defaults to configured model for provider.

        Returns:
            LLM instance (ChatOllama or ChatOpenAI)
        """
        current_provider = provider or cls.MODEL_PROVIDER

        if current_provider == "openai":
            if not cls.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")

            current_model = model or cls.OPENAI_MODEL
            return ChatOpenAI(
                model=current_model,
                temperature=cls.OPENAI_TEMPERATURE,
                max_tokens=cls.OPENAI_MAX_TOKENS,
                openai_api_key=cls.OPENAI_API_KEY
            )

        elif current_provider == "ollama":
            current_model = model or cls.OLLAMA_MODEL
            return ChatOllama(
                model=current_model,
                temperature=cls.OLLAMA_TEMPERATURE,
                base_url=cls.OLLAMA_BASE_URL
            )

        else:
            raise ValueError(f"Unsupported model provider: {current_provider}. Use 'ollama' or 'openai'")

    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present."""
        # Validate input file
        possible_paths = [
            cls.INPUT_FILE,  # dataset/dev_top20_matches.pkl
            "dev_top20_matches.pkl",  # root directory fallback
            os.path.join("dataset", "dev_top20_matches.pkl")  # explicit dataset path
        ]

        for path in possible_paths:
            if os.path.exists(path):
                cls.INPUT_FILE = path  # Update to the found path
                print(f"✅ Found input file at: {path}")
                break
        else:
            raise FileNotFoundError(f"Input file not found in any of these locations: {possible_paths}")

        # Validate provider-specific requirements
        if cls.MODEL_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required when using OpenAI models")

        print(f"✅ Configuration validated for provider: {cls.MODEL_PROVIDER}")

        return True

    @classmethod
    def list_supported_models(cls):
        """List all supported models by provider."""
        models = {
            "ollama": [
                "gemma3:1b-it-fp16",
                "llama3.1:8b",
                "llama3.1:70b",
                "qwen2.5:7b",
                # Add other Ollama models as needed
            ],
            "openai": [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ]
        }
        return models

    @classmethod
    def print_config_summary(cls):
        """Print a summary of current configuration."""
        print("=" * 50)
        print("Islamic Text Verification System - Configuration")
        print("=" * 50)
        print(f"Provider: {cls.MODEL_PROVIDER}")

        if cls.MODEL_PROVIDER == "openai":
            print(f"OpenAI Model: {cls.OPENAI_MODEL}")
            print(f"OpenAI Temperature: {cls.OPENAI_TEMPERATURE}")
            print(f"OpenAI Max Tokens: {cls.OPENAI_MAX_TOKENS}")
            print(f"API Key: {'✅ Set' if cls.OPENAI_API_KEY else '❌ Missing'}")
        else:
            print(f"Ollama Model: {cls.OLLAMA_MODEL}")
            print(f"Ollama Temperature: {cls.OLLAMA_TEMPERATURE}")
            print(f"Ollama Base URL: {cls.OLLAMA_BASE_URL}")

        print(f"Remove Diacritics: {cls.REMOVE_DIACRITICS}")
        print(f"Input File: {cls.INPUT_FILE}")
        print(f"Results Folder: {cls.get_results_folder()}")
        print("=" * 50)
