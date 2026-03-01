"""
Verse merging utilities for combining consecutive Quranic verses.
"""

from typing import List, Dict, Any


def merge_ayas_from_retrieval(verses_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups and merges consecutive verses from the same surah.

    Args:
        verses_data: List of dictionaries containing verse information

    Returns:
        List of merged verse groups, where consecutive verses are combined
    """
    if not verses_data:
        return []

    # Parse verse_id and add surah and ayah numbers for sorting
    parsed_verses = []
    for verse in verses_data:
        verse_id = verse['verse_id']
        try:
            surah_num, ayah_num = map(int, verse_id.split(':'))
            verse_copy = verse.copy()
            verse_copy['surah_num'] = surah_num
            verse_copy['ayah_num'] = ayah_num
            parsed_verses.append(verse_copy)
        except (ValueError, IndexError):
            # If parsing fails, treat as individual verse
            parsed_verses.append(verse)

    # Sort by surah number, then by ayah number
    parsed_verses.sort(key=lambda x: (x.get('surah_num', 0), x.get('ayah_num', 0)))

    # Group consecutive verses from same surah
    groups = []
    current_group = [parsed_verses[0]]

    for i in range(1, len(parsed_verses)):
        current_verse = parsed_verses[i]
        last_verse = current_group[-1]

        # Check if consecutive and same surah
        if (current_verse.get('surah_num') == last_verse.get('surah_num') and
            current_verse.get('ayah_num') == last_verse.get('ayah_num', 0) + 1):
            current_group.append(current_verse)
        else:
            groups.append(current_group)
            current_group = [current_verse]

    groups.append(current_group)

    # Create merged entries for each group
    merged_results = []
    for group in groups:
        merged_entry = create_merged_entry(group)
        merged_results.append(merged_entry)

    return merged_results


def create_merged_entry(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates a merged entry from a group of consecutive verses.

    Args:
        group: List of consecutive verses from same surah

    Returns:
        Dictionary representing the merged group
    """
    if len(group) == 1:
        # Single verse - return as is but clean up temporary fields
        verse = group[0].copy()
        verse.pop('surah_num', None)
        verse.pop('ayah_num', None)
        return verse

    # Multiple verses - merge them
    first_verse = group[0]
    last_verse = group[-1]

    # Create range for verse_id (e.g., "2:155-157")
    if first_verse['ayah_num'] == last_verse['ayah_num']:
        merged_verse_id = first_verse['verse_id']
    else:
        merged_verse_id = f"{first_verse['surah_num']}:{first_verse['ayah_num']}-{last_verse['ayah_num']}"

    # Combine ayah texts
    combined_text = ' '.join([verse['ayah_text'] for verse in group])

    # Use the highest similarity score in the group
    max_similarity = max([verse['similarity_score'] for verse in group])

    merged_entry = {
        'verse_id': merged_verse_id,
        'similarity_score': max_similarity,
        'surah_name': first_verse['surah_name'],
        'ayah_text': combined_text,
        'merged_count': len(group),
        'original_verses': [verse['verse_id'] for verse in group]
    }

    return merged_entry
