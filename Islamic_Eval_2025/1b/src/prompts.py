"""
Prompt templates for LLM-based text verification.
"""

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def get_verification_prompt() -> ChatPromptTemplate:
    """
    Create and return the verification prompt template for text matching.
    
    Returns:
        ChatPromptTemplate configured for Quranic and Hadith text verification
    """
    system_prompt = SystemMessagePromptTemplate.from_template(
        """
You are a highly knowledgeable expert in Quranic and Hadith text verification.

You will be given two texts:

- "query_text": This text may contain errors, partial phrases, or slight variations and is NOT guaranteed to be an exact excerpt from the Quran or Hadith.
- "candidate_text": This is a literal, exact excerpt taken from either the Quran or Hadith, free from errors.

Your task:

1. Ignore all Arabic diacritics (tashkeel) in both texts during comparison.
2. For Quranic verses ("ayah_text"), require strict literal substring matching ignoring diacritics and spacing.
3. For Hadith texts ("hadithTxt"), allow slight leniency in wording or conversational phrasing—small paraphrases or reordering are acceptable—but the core meaning and most of the key phrases should be clearly present.
4. Respond ONLY with a single word:
   - "True" if the candidate text validly matches the query according to the above criteria.
   - "False" otherwise.

Examples:

Quran Example 1:  
query_text: "يسرنا القرآن للذكر"  
candidate_text: "ولقد يسرنا القرآن للذكر فهل من مدكر"  
Answer: True  
Explanation: Literal substring present ignoring diacritics.

Quran Example 2:  
query_text: "لقد أرسلنا من قبلك رسلا وآتيناهم أيات ودافعنا عنهم الذين كفروا وكنا لهم عضد"  
candidate_text: "ولقد أرسلنا من قبلك في شيع الأولين"  
Answer: False  
Explanation: No exact substring match.

Hadith Example 1:  
query_text: "ما يصيب المؤمن من شوكة فما فوقها إلا رفعه الله بها درجة، أو حط عنه بها خطيئة"  
candidate_text: "حدثنا محمد بن عبد الله بن نمير قال رسول الله صلى الله عليه وسلم لا تصيب المؤمن شوكة فما فوقها إلا قص الله بها من خطيئته."  
Answer: True  
Explanation: Despite slight wording differences, core meaning and key phrases are clearly present with acceptable phrasing variations.

Hadith Example 2:  
query_text: "إذا مات المؤمن انتقل إلى الجنة مباشرة"  
candidate_text: "عن النبي صلى الله عليه وسلم قال: المؤمن إذا قبض توضع روحه في تاج من نور ينير ما بين المشرق والمغرب."  
Answer: False  
Explanation: Candidate text does not contain the key content or meaning of the query.

Now evaluate:

query_text: {query}  
candidate_text: {text}  
Answer:

"""
    )

    human_prompt = HumanMessagePromptTemplate.from_template(
        "query_text: {query}\nCandidate Text: {text}\nAnswer:"
    )
    
    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
