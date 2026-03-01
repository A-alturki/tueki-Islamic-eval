Subtasks 1A, B, and 1C:
Hallucination Detection and Correction
Task Definition
Task Definition
Example
Evaluation Measures
Registration
Dataset
Run Submission
Task Definition
The subtasks are defined as follows: Given a free-text response from an LLM to a religious question, the answer might or might not contain Quranic verses (Ayahs) and Hadiths (Prophetic sayings). These Ayahs and Hadiths, if any, might be inaccurate due to LLM hallucination. For this free-text response, it is required to identify each claimed --since they might be inaccurate-- Ayah and Hadith (subtask 1A), identify if each Ayah or Hadith is correct or not (subtask 1B), and identify the corrected version of the Ayah and the Hadith if any (subtask 1C).

More specifically, the three subtasks are:

Subtask 1A: Identification of Intended Ayahs (Quranic verses) and Hadiths (Prophetic sayings). Given an LLM-generated response, participants will determine the spans of the "intended" Ayahs and Hadiths included in the text. Spans are represented by the character indexes, ex: from character 0 until character 72.  Evaluation will be based on span precision and recall.  References to Ayah number and Hadith narrators and punctuations will be ignored in this version of the shared task.


Subtask 1B: Validation of content accuracy. For each identified Ayah and Hadith, participants will categorize them as correct or incorrect based on established Islamic references. Evaluation will be based on precision and recall.  Incorrect diacritics will be considered mistakes.


Subtask 1C: Correction of Erroneous Content. Participants will provide corrected versions for any incorrectly generated Ayah or Hadith, ensuring fidelity to the original sources. Evaluation will be based on Word Error Rate (WER). Note that complete Ayahs and complete Hadiths are expected. Writing and diacritics of Ayahs and Hadiths should be obtained from the shared Quran and Hadith sources.

Task Definition
The subtasks are defined as follows: Given a free-text response from an LLM to a religious question, the answer might or might not contain Quranic verses (Ayahs) and Hadiths (Prophetic sayings). These Ayahs and Hadiths, if any, might be inaccurate due to LLM hallucination. For this free-text response, it is required to identify each claimed --since they might be inaccurate-- Ayah and Hadith (subtask 1A), identify if each Ayah or Hadith is correct or not (subtask 1B), and identify the corrected version of the Ayah and the Hadith if any (subtask 1C).

More specifically, the three subtasks are:

Subtask 1A: Identification of Intended Ayahs (Quranic verses) and Hadiths (Prophetic sayings). Given an LLM-generated response, participants will determine the spans of the "intended" Ayahs and Hadiths included in the text. Spans are represented by the character indexes, ex: from character 0 until character 72.  Evaluation will be based on span precision and recall.  References to Ayah number and Hadith narrators and punctuations will be ignored in this version of the shared task.


Subtask 1B: Validation of content accuracy. For each identified Ayah and Hadith, participants will categorize them as correct or incorrect based on established Islamic references. Evaluation will be based on precision and recall.  Incorrect diacritics will be considered mistakes.


Subtask 1C: Correction of Erroneous Content. Participants will provide corrected versions for any incorrectly generated Ayah or Hadith, ensuring fidelity to the original sources. Evaluation will be based on Word Error Rate (WER). Note that complete Ayahs and complete Hadiths are expected. Writing and diacritics of Ayahs and Hadiths should be obtained from the shared Quran and Hadith sources.


Example
In the example above, the required from each each subtask is as follows:

Subtask 1A:
Identify the text spans of two claimed Ayahs: "قُل لَّا يَعْلَمُ مَن فِي السَّمَاوَاتِ وَالْأَرْضِ الْغَيْبَ إِلَّا اللَّهُ" and "وَمَا كَانَ لِسْلَيْمَنَ نَفَقَةُ وَلَا هُوَ يَعْلَمُ مَا فِيَ غَيْبِ السَّمُوُتِ وَالْأَرْض وَٱللَّهُ عَلِمُ بِمَا تَعْمَلُونَ", and one claimed Hadith "مفاتح الغيب خمس لا يعلمهن إلا الله"

Subtask 1B:
Identify that Ayah1 is correct, Ayah2 is incorrect, and Hadith1 is incorrect.

Subtask 1C:
Provide the correction for the claimed Hadith, which should be "مفاتح الغيب خمس لا يعلمها إلا الله", and there is no clear correction for the claimed Ayah2.

Evaluation Measures
Subtask 1A: 

Macro-Averaged F1 Score.  F1 Score will be computed at the character level by treating each character (i.e. index) of the response string as either Ayah/Hadith/Neither. 

Subtask 1B: 

Accuracy. Number of accurate labels (Correct/Incorrect) divided by number of total labels.

Subtask 1C: 

Accuracy. Number of accurate corrections divided by number of total corrections.

Registration
The registration form is available at: Form

Dataset
The development set for Subtasks 1A, 1B, and 1C is released: Link.

The file also has the texts for Quran and Hadith (from the six books الكتب الستة) in JSON format.  You are strictly required to use the data provided.

Sources of Quran (in Othmani script) and Hadith are: Quran, Hadith.  

Run Submission
Please submit your runs on the following links:

Subtask 1A: https://www.codabench.org/competitions/9820/

Subtask 1B: https://www.codabench.org/competitions/9822/

Subtask 1C: https://www.codabench.org/competitions/9824/

The submission format for each subtask is described in detail in the subtask page.

