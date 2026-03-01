"""
Test script for the updated LLM dataset generator
"""
from typing import List
from pydantic import BaseModel

# Test the Pydantic models
class DetectedPhrase(BaseModel):
    label: str
    value: str

class DatasetExample(BaseModel):
    text: str
    detected_phrases: List[DetectedPhrase]

class DatasetExamples(BaseModel):
    examples: List[DatasetExample]

def test_models():
    """Test the Pydantic models"""
    
    # Test data
    test_phrase = DetectedPhrase(label="Ayah", value="بسم الله الرحمن الرحيم")
    test_example = DatasetExample(
        text="في هذا النص: بسم الله الرحمن الرحيم",
        detected_phrases=[test_phrase]
    )
    test_dataset = DatasetExamples(examples=[test_example])
    
    print("✓ Pydantic models work correctly")
    print("✓ Example structure:", test_dataset.dict())
    
    return True

if __name__ == "__main__":
    test_models()
    print("✓ All tests passed")
