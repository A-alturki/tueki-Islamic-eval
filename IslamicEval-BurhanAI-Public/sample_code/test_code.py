from openai import OpenAI
client = OpenAI()

response = client.responses.create(
  model="gpt-5",
  input=[
    {
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": "Extract the list of citations from the text based on the schema, please use code interpretation if needed to get the accurate span start and end.\n\nBefore giving the final answer - please insure that you check the file search tool (the full quran and hadeeth is included) - to get the result with utmost accuracy, and use web search for a final measure (only use most trusted sources)\n\nThe same text of the question is attached to the code interpreter with file name: text_to_analyse.txt to make it easy to analyze via code."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "........"
        }
      ]
    }
  ],
  text={
    "format": {
      "type": "json_schema",
      "name": "citations_validation",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "citations": {
            "type": "array",
            "description": "List of citations with details, validation, references, and per-citation corrections.",
            "items": {
              "type": "object",
              "properties": {
                "span_start": {
                  "type": "integer",
                  "description": "Inclusive start character index of the citation within the text (exclude the qutation marks)."
                },
                "span_end": {
                  "type": "integer",
                  "description": "Exclusive end character index of the citation within the text (exclude the qutation marks)."
                },
                "span_text": {
                  "type": "string",
                  "description": "The text content of the citation span."
                },
                "citation_type": {
                  "type": "string",
                  "description": "Type of the citation: either Aya (Quran) or Hadeeth (Prophetic tradition).",
                  "enum": [
                    "Aya",
                    "Hadeeth"
                  ]
                },
                "is_valid": {
                  "type": "boolean",
                  "description": "True if the cited Aya or Hadeeth is recognized and valid, false otherwise."
                },
                "aya_reference": {
                  "type": "object",
                  "description": "Quranic reference details (required if citation_type is Aya; empty object if not applicable).",
                  "properties": {
                    "sura_name": {
                      "type": "string",
                      "description": "Name of the sura (chapter) in the Quran."
                    },
                    "sura_number": {
                      "type": "integer",
                      "description": "Number of the sura (chapter) in the Quran."
                    },
                    "aya_number": {
                      "type": "integer",
                      "description": "Number of the aya (verse) in the Quran."
                    }
                  },
                  "required": [
                    "sura_name",
                    "sura_number",
                    "aya_number"
                  ],
                  "additionalProperties": False
                },
                "hadeeth_reference": {
                  "type": "object",
                  "description": "Hadeeth reference details (required if citation_type is Hadeeth; empty object if not applicable).",
                  "properties": {
                    "collection_name": {
                      "type": "string",
                      "description": "Name of the Hadeeth collection in english (e.g. Sahih Bukhari, Muslim)."
                    },
                    "book_number": {
                      "type": "integer",
                      "description": "Book number within the collection."
                    },
                    "hadith_number": {
                      "type": "integer",
                      "description": "Hadith number within the specified book."
                    }
                  },
                  "required": [
                    "collection_name",
                    "book_number",
                    "hadith_number"
                  ],
                  "additionalProperties": False
                },
                "corrected_text": {
                  "type": "string",
                  "description": "The corrected version of the citation text (or the text itself if it was correct in the first place) (e.g. if there are certain words that were invalid this version must have this corrected), or 'خطأ' for completely invalid citations (i.e. no close enough citation can be found to correct the text). If there are close texts in Quran or Hadith - but too far from the citation, this must be considered 'خطأ' and completely_invalid must be true."
                },
                "completely_invalid": {
                  "type": "boolean",
                  "description": "Set to true if the citation is completely invalid i.e. there can't be a close enough text to correct it, (corrected_text must be 'خطأ')."
                }
              },
              "required": [
                "span_start",
                "span_end",
                "span_text",
                "citation_type",
                "is_valid",
                "aya_reference",
                "hadeeth_reference",
                "corrected_text",
                "completely_invalid"
              ],
              "additionalProperties": False
            }
          }
        },
        "required": [
          "citations"
        ],
        "additionalProperties": False
      }
    },
    "verbosity": "high"
  },
  reasoning={
    "effort": "high",
    "summary": null
  },
  tools=[
    {
      "type": "file_search",
      "vector_store_ids": [
        "vs_688f4806b52081919c61c4a541be1dbb"
      ]
    },
    {
      "type": "web_search_preview",
      "user_location": {
        "type": "approximate"
      },
      "search_context_size": "medium"
    },
    {
      "type": "code_interpreter",
      "container": {
        "type": "auto",
        "file_ids": [
          "file-1uR8HmwEnLRyLYMZMRPDwf"
        ]
      }
    }
  ],
  store=False
)
