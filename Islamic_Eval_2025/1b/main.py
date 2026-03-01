#!/usr/bin/env python3
"""
Main entry point for the Islamic Text Verification System.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.verifier import IslamicTextVerifier


def main():
    """Main function to run the Islamic Text Verification System."""
    try:
        verifier = IslamicTextVerifier()
        verifier.run()
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
