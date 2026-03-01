"""
Text processing utilities for Arabic text cleaning and normalization.
"""

import pyarabic.araby as araby
from .config import Config


def clean_text(text: str, remove_diacritics: bool = None) -> str:
    """
    Clean Arabic text by removing diacritics (tashkeel).
    
    Args:
        text: Raw Arabic text string
        remove_diacritics: Whether to remove diacritics. If None, uses config setting.
        
    Returns:
        Cleaned text string with diacritics removed if enabled
    """
    if not text:
        return ""

    # Use config setting if not explicitly provided
    if remove_diacritics is None:
        remove_diacritics = Config.REMOVE_DIACRITICS

    # Remove diacritics (tashkeel) if flag is enabled
    if remove_diacritics:
        clean_text = araby.strip_tashkeel(text)
        return clean_text
    
    return text
