import json
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Set

import pyquran as q
from pyarabic.araby import strip_tashkeel
from pyarabic.normalize import strip_tatweel, normalize_lamalef


class QuranSearchEngine:
    """Enhanced Quranic text search engine with improved accuracy and performance."""

    def __init__(self, authentic_quran_file_path: str = None, ):
        """
        Initialize the search engine.

        Args:
            authentic_quran_file_path: Path to the authentic Quran JSON file with new format

        """
        # Load authentic Quran data
        self.authentic_quran_data = self._load_authentic_quran_data(authentic_quran_file_path)

    def _load_authentic_quran_data(self, file_path: str) -> dict:
        """Load authentic Quran data from JSON file with new format."""
        try:
            if not file_path or not os.path.exists(file_path):
                print(f"Warning: Authentic Quran file not found: {file_path}")
                return {}

            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Create a lookup dictionary for quick access
            # Key format: "surah_id:ayah_id" -> ayah_data
            lookup = {}
            surah_names = {}  # surah_id -> surah_name
            surah_names_en = {}  # surah_id -> English name (if available)

            for item in data:
                key = f"{item['surah_id']}:{item['ayah_id']}"
                lookup[key] = {
                    'ayah_text': item['ayah_text'],
                    'surah_name': item['surah_name']
                }
                # Store surah names for quick lookup
                surah_names[item['surah_id']] = item['surah_name']
                # Try to get English name if available in the data
                surah_names_en[item['surah_id']] = item.get('surah_name_en', '')

            return {
                'lookup': lookup,
                'surah_names': surah_names,
                'surah_names_en': surah_names_en
            }

        except Exception as e:
            print(f"Error loading authentic Quran data: {e}")
            return {}

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean Arabic text by removing diacritics and normalizing characters.

        Args:
            text: Raw Arabic text

        Returns:
            Cleaned Arabic text
        """
        if not text:
            return ""

        text = strip_tashkeel(text)
        text = strip_tatweel(text)
        text = normalize_lamalef(text)
        return text.strip()

    def _get_authentic_verse_text(self, sura_number: int, verse_number: int) -> str:
        """
        Get authentic verse text from the JSON file.

        Args:
            sura_number: Surah number
            verse_number: Verse number

        Returns:
            Authentic verse text or empty string if not found
        """
        if not self.authentic_quran_data:
            print(f"Warning: No authentic Quran data loaded")
            return ""

        try:
            key = f"{sura_number}:{verse_number}"
            if key in self.authentic_quran_data.get('lookup', {}):
                return self.authentic_quran_data['lookup'][key]['ayah_text']
            else:
                print(f"Warning: Verse {sura_number}:{verse_number} not found in authentic data")
                return ""
        except Exception as e:
            print(f"Error getting authentic verse text for {sura_number}:{verse_number}: {e}")
            return ""

    def _get_authentic_surah_name(self, sura_number: int) -> str:
        """
        Get authentic surah name from the JSON file.

        Args:
            sura_number: Surah number

        Returns:
            Authentic surah name or empty string if not found
        """
        return self.authentic_quran_data.get('surah_names', {}).get(sura_number, '')

    def _get_authentic_surah_name_en(self, sura_number: int) -> str:
        """
        Get authentic surah English name from the JSON file.

        Args:
            sura_number: Surah number

        Returns:
            Authentic surah English name or empty string if not found
        """
        return self.authentic_quran_data.get('surah_names_en', {}).get(sura_number, '')

    def search_verse_matches(self, search_query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for verses containing words from the query.

        Args:
            search_query: Arabic text to search for
            max_results: Maximum number of results to return (default: 10)

        Returns:
            List of dictionaries containing verse information and match details
        """
        if not search_query.strip():
            return []

        # Clean and prepare the search query
        cleaned_query = self.clean_text(search_query)
        words = [word for word in cleaned_query.split() if word]

        if not words:
            return []

        # Track word occurrences in verses
        verse_word_matches = defaultdict(set)

        # Search for each word individually
        for word_index, word in enumerate(words):
            try:
                search_results = q.search_sequence(sequancesList=[word], mode=2)

                for sequence, matches in search_results.items():
                    if matches:
                        for match in matches:
                            matched_text, position_in_verse, verse_num, chapter_num = match
                            verse_key = (chapter_num, verse_num)
                            verse_word_matches[verse_key].add((sequence, word_index, position_in_verse))

            except Exception as e:
                print(f"Search failed for word '{word}': {e}")
                continue

        return self._rank_and_format_results(verse_word_matches, words, max_results)

    def _rank_and_format_results(self, verse_matches: Dict, query_words: List[str], max_results: int) -> List[Dict]:
        """
        Rank verses by relevance and format results.

        Args:
            verse_matches: Dictionary of verse matches
            query_words: List of search words
            max_results: Maximum number of results to return

        Returns:
            Ranked list of verse matches with match information
        """
        if not verse_matches:
            return []

        qualified_verses = {
            verse_key: matches
            for verse_key, matches in verse_matches.items()
        }

        results = []

        for verse_key, word_matches in qualified_verses.items():
            sura_number, verse_number = verse_key

            # Calculate word proximity score (lower is better)
            word_positions = [match[2] for match in word_matches]  # position_in_verse
            proximity_score = self._calculate_proximity_score(word_positions)

            # Calculate coverage score (percentage of query words found)
            unique_word_indices = set(match[1] for match in word_matches)  # word_index
            coverage_score = len(unique_word_indices) / len(query_words)

            # Combined score: prioritize coverage, then proximity
            final_score = (coverage_score * 1000) - proximity_score

            try:
                # Get verse text from authentic JSON file
                verse_text = self._get_authentic_verse_text(sura_number, verse_number)

                # Skip if verse text not found
                if not verse_text:
                    continue

                # Get surah names from authentic data
                surah_name_ar = self._get_authentic_surah_name(sura_number)
                surah_name_en = self._get_authentic_surah_name_en(sura_number)

                # Create result dictionary
                result = {
                    'score': final_score,
                    'surah_number': sura_number,
                    'verse_number': verse_number,
                    'verse_text': verse_text,
                    'surah_name_ar': surah_name_ar,
                    'surah_name_en': surah_name_en,
                    'coverage_score': coverage_score,
                    'proximity_score': proximity_score,
                    'matched_words_count': len(word_matches),
                    'total_query_words': len(query_words),
                }

                results.append(result)

            except Exception as e:
                print(f"Error retrieving verse {sura_number}:{verse_number}: {e}")
                continue

        # Sort by score (descending) and return top results
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:max_results]

    @staticmethod
    def _calculate_proximity_score(positions: List[int]) -> float:
        """
        Calculate how close words are to each other in the verse.

        Args:
            positions: List of word positions in the verse

        Returns:
            Proximity score (lower means words are closer together)
        """
        if len(positions) <= 1:
            return 0

        sorted_positions = sorted(positions)
        total_distance = sum(
            sorted_positions[i + 1] - sorted_positions[i]
            for i in range(len(sorted_positions) - 1)
        )
        return total_distance / (len(positions) - 1)


def main():
    """Main function to test the search engine with multiple queries."""

    # Test queries
    test_queries = [
        "َلَقَدْ أَرْسَلْنَا مِن قَبْلِكَ رُسُلًا وَآتَيْنَاهُمْ أَيَاتٍ وَدَافَعْنَا عَنْهُمْ الَّذِينَ كَفَرُوا وَكُنَّا لَهُمْ عَضُدً",
        "وَلَقَدْ أَرْسَلْنَا مِن قَبْلِكَ رُسُلًا إِلَى قَوْمٍ مِنْهُمْ لَمْ تُكَلُفْ بَعْضُهُمْ بَعْضً",
        'حُجَّتُنَآ ءَاتَيْنَاهَآ إِبْرَٰهِيمَ عَلَىٰ قَوْمِهِ',
        'يَسَّرْنَا الْقُرْآنَ لِلذِّكْرِ',
        'وَتِلْكَ حُجَّتُنَآ ءَاتَيْنَاهَآ إِبْرَٰهِيمَ عَلَىٰ قَوْمِهِۦۚ نَرۡفَعُ دَرَجَٰتٖ مَّن نَّشَآءُۗ إِنَّ رَبَّكَ حَكِيمٌ عَلِيمٌ',
        'وأَفَلَا يَتَدَبَّرُونَ الْقُرْآنَ أَمْ عَلَىٰ قُلُوبٍ أَقْفَالُهَا',
        'كِتَابٌ أَنزَلْنَاهُ إِلَيْكَ مُبَارَكٌ لِيَدَّبَّرُوا آيَاتِهِ وَلِيَتَذَكَّرَ أُولُو الْأَلْبَابِ',
        'وَلَقَدْ يَسَّرْنَا الْقُرْآنَ لِلذِّكْرِ فَهَلْ مِن مُّدَّكِرٍ',
        'الَّذِينَ يَسْتَمِعُونَ الْقَوْلَ فَيَتَّبِعُونَ أَحْسَنَهُ، أُولَٰئِكَ الَّذِينَ هَدَاهُمُ اللَّهُ وَأُولَٰئِكَ هُمْ أُولُو الْأَلْبَابٍِ',
        "وإذ قال ربك للملائكة إني جاعل في الأرض خليفة ۖ قالوا أتجعل فيها من يفسد فيها ويسفك الدماء ونحن نسبح بحمدك ونقدس لك ۖ قال إني أعلم ما لا تعلمون",
        "وإذ قلتم يا موسى لن نصبر على طعام واحد فادع لنا ربك يخرج لنا مما تنبت الأرض من بقلها وقثائها وفومها وعدسها وبصلها ۖ قال أتستبدلون الذي هو أدنى بالذي هو خير ۚ اهبطوا مصرا فإن لكم ما سألتم ۗ وضربت عليهم الذلة والمسكنة وباءوا بغضب من الله ۗ ذلك بأنهم كانوا يكفرون بآيات الله ويقتلون النبيين بغير الحق ۗ ذلك بما عصوا وكانوا يعتدون",
        "إذ قال له ربه أسلم ۖ قال أسلمت لرب العالمين",
        "فإن انتهوا فإن الله غفور رحيم",
        "ولما برزوا لجالوت وجنوده قالوا ربنا أفرغ علينا صبرا وثبت أقدامنا وانصرنا على القوم الكافرين",
        "أفرغ علينا صبرا وثبت أقدامنا وانصرنا على القوم ",
        "دْ كَانَ لَكُمْ فِي رَسُولِ اللَّهِ أُسْوَةٌ حَسَنَةٌ لِمَن كَانَ يَرْجُو اللَّهَ وَالْيَوْمَ الْآخِرَ وَذَكَرَ اللَّهَ",
        "نُفِخَ فِي الصُّورِ فَإِذَا هُمْ مِنَ الْأَجْدَاثِ إِلَىٰ رَبِّهِمْ يَنسِلُونَ",
        "وَلَقَدْ أَرْسَلْنَا إِلَيْهِمْ رَسُولًا مِنْهُمْ فَقَالُوا لَوْ يَطَاعُ مِنْهُمْ لَأَطَعْنَا",
        "وَلَا تُصَعِّرْ خَدَّكَ لِلنَّاسِ وَلَا تَمْشِ فِي الْأَرْضِ مَرَحًا إِنَّ اللَّهَ لَا يُحِبُّ كُلَّ مُخْتَالٍ فَخُورٍ (١٨) وَاقْصِدْ فِي مَشْيِكَ وَاغْضُض",
        "من يقاتل في سبيل الله فيقتل أو يغلب فسوف يؤتيه أجرا عظيما",
        "وَلَا تُسْرِفْ فِي الْأَمْوَالِ",
        "قَالُوا ادْعُ لَنَا رَبَّكَ يُبَيِّنْ لَنَا مَا هِيَ قَالَ إِنَّهُ يَقُولُ إِنَّهَا بَقَرَةٌ لَا ذَلُولٌ تُثِيرُ الْأَرْضَ وَلَا تَسْقِي الْحَرْثَ مُسَلَّمَةٌ لَا شِيَةَ فِيهَا قَالُوا الْآنَ جِئْتَ بِالْحَقِّ فَذَبَحُوهَا وَمَا كَادُوا يَفْعَلُونَ",
        "المحرَمات من النساء في الزواج هن الأخوات، والشُرَّعات، والأُمَّهات، والأَخوات، والشُرَّعات، والأُختان اللتان ترضينَ منهنَّ شيئاً"
    ]

    # Initialize the search engine with your JSON file
    authentic_quran_file_path = "resources/quranic_verses/quranic_verses.json"

    search_engine = QuranSearchEngine(
        authentic_quran_file_path=authentic_quran_file_path,
    )

    # Test each query and display results
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Query {i}: {query}")
        print(f"{'=' * 80}")

        try:
            results = search_engine.search_verse_matches(query, max_results=10)

            if results:
                print(f"Found {len(results)} matching verses:")
                print("-" * 80)

                for j, result in enumerate(results, 1):
                    print(f"{j}. Surah {result['surah_number']} ({result['surah_name_ar']}) - Verse {result['verse_number']}")
                    print(f"   Score: {result['score']:.2f} | Coverage: {result['coverage_score']:.2f} | Proximity: {result['proximity_score']:.2f}")
                    print(f"   Text: {result['verse_text']}")
                    print()
            else:
                print("No matching verses found.")

        except Exception as e:
            print(f"Search failed: {e}")

    print(f"\n{'=' * 80}")
    print("Testing completed.")


if __name__ == "__main__":
    main()
