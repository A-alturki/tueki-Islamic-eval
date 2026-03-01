
IslamicEval 2025 Shared Task!
Home
Subtask 1
Subtask 2
Important Dates
FAQ
Welcome to IslamicEval 2025 Shared Task!
Capturing LLMs Hallucination in Islamic Content
@ArabicNLP 2025, co-located with EMNLP 2025

China, November 2025

Recent Updates
29 July 2025: Test sets for Subtasks 1A, 1B, 1C, and 2, and the CodaBench competitions are all available. 

16 June 2025: Registration Form is available. 

16 June 2025: Data is released: 

Subtasks 1A, 1B, and 1C: The development set is released. It includes the texts for Quran and Hadith (from the six books الكتب الستة) in JSON format.  You are strictly required to use the data provided. Sources of Quran (in Othmani script) and Hadith are: Quran, Hadith.

Subtask 2: The training and development sets are released. The repo also includes the collection of the Qur'anic passages and the collection of Hadiths for this subtask. Find more details here.

16 June 2025: Website is up! 

28 April 2025: IslamicEval 2025 proposal accepted!



Information Hub!
All details about the task and how it is evaluated can be found on the Subtask 1A, 1B, 1C, and Subtask 2 pages.

Do you have any questions about the task? Check our FAQ page or post it on our Discussion Group.

Discussion Group
Please join our Google group at https://groups.google.com/g/islamic-eval to receive announcements and participate in discussions.


Why IslamicEval?
Large Language Models (LLMs) are becoming an integral part of natural language processing applications in Arabic. Recent advancements have produced several Arabic-focused and multilingual LLMs, such as Jais, Allam, and Fanar, which have shown promising results across a variety of tasks, from open-domain question answering to content generation. However, alongside these advances, a critical challenge remains unresolved, namely hallucination, i.e. the generation of text that appears plausible but is factually incorrect or fabricated.

This issue is particularly sensitive in domains where accuracy and authenticity are paramount, such as religion. In the Arabic-speaking world, religious topics are culturally central and frequently searched, discussed, and queried on online platforms and social media. This makes religious discourse one of the most common applications for Arabic LLMs, whether directly or indirectly.

Among religious sources, the Quran and Hadith literature stand out due to their sacred status and the high expectations of precision when they are quoted or referenced. Unfortunately, hallucination in LLMs can result in misattributions, paraphrased verses falsely labeled as genuine, or entirely fabricated Hadiths, which raises serious ethical, theological, and social concerns. Such hallucinations can unintentionally propagate misinformation or be exploited for disinformation, undermining trust in AI technologies and amplifying harm.


Task Overview
The first version of our IslamicEval shared task tackles hallucination in LLMs at two stages with four mainly complementary subtasks. Given a question, the subtasks target the identification of Islamic content, the validation of that content, the correction of it, and then deciding if it is actually relevant to the given question or not.


Subtask 1A: Identification of Intended Ayahs (Quranic verses) and Hadiths (Prophetic sayings). Given an LLM-generated response, participants will determine the spans of the "intended" ayahs and hadiths included in the text. Spans are represented by the character indexes, ex: from character 0 until character 72.  Evaluation will be based on span precision and recall.  References to Ayah number and Hadith narrators and punctuations will be ignored.


Subtask 1B: Validation of content accuracy. For each identified Ayah and Hadith, participants will categorize them as correct or incorrect based on established Islamic references. Evaluation will be based on precision and recall.  Incorrect diacritics will be considered as mistakes.


Subtask 1C: Correction of Erroneous Content. Participants will provide corrected versions for any incorrectly generated Ayah or Hadith, ensuring fidelity to the original sources. Evaluation will be based on Word Error Rate. Note that complete verse(s) from Quran and complete Hadiths are expected. Writing and diacritics of Quran verses and Hadiths should be obtained from the shared Quran and Hadith sources.


Subtask 2: Qur'an and Hadith QA. Given a free-text question posed in MSA, a collection of Qur'anic passages that cover the Holy Qur'an, and a collection of Hadiths from Sahih Bukhari, a system is required to return a ranked list of answer-bearing qur'anic passages or Hadiths from the two collections. 

All details about the tasks and how they are evaluated can be found on Subtask 1A, 1B, 1C, and Subtask 2  pages.

Important Dates
All times are Anywhere On Earth (AOE).

16 June 2025: Data is released

16 June 2025: Registration opens

Subtasks

29 July 2025: Test set is released

6 August 2025: Registration closes

8 August 2025: Submission on test set closes

10 August 2025: Results are sent to teams

Paper Submissions

15 August 2025: Shared-task paper submission deadline

25 August 2025: Notification of acceptance

5 September 2025: Camera-ready papers due

Conference

November 5-9, 2025: Main Conference





Organizers
Hamdy Mubarak, QCRI, HBKU

Abubakr Mohamed, QCRI, HBKU

Majd Hawasly, QCRI, HBKU

Walid Magdy, University of Edinburgh

Kareem Darwish, QCRI, HBKU

Rana Malhas, Qatar University

Watheq Mansour, University of Queensland

Tamer Elsayed, Qatar University

Mahmoud Fawzi, University of Edinburgh

Contact

