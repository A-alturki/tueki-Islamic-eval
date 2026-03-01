Task 1 Def:
The subtasks are defined as follows: Given a free-text response from an LLM to a religious question, the answer might or might not contain Quranic verses (Ayahs) and Hadiths (Prophetic sayings). These Ayahs and Hadiths, if any, might be inaccurate due to LLM hallucination. For this free-text response, it is required to identify each claimed --since they might be inaccurate-- Ayah and Hadith (subtask 1A), identify if each Ayah or Hadith is correct or not (subtask 1B), and identify the corrected version of the Ayah and the Hadith if any (subtask 1C).



Subtask 1C: Correction of Erroneous Content. Participants will provide corrected versions for any incorrectly generated Ayah or Hadith, ensuring fidelity to the original sources. Evaluation will be based on Word Error Rate (WER). Note that complete Ayahs and complete Hadiths are expected. Writing and diacritics of Ayahs and Hadiths should be obtained from the shared Quran and Hadith sources.

Subtask 1C: Correction of Erroneous Content. Participants will provide corrected versions for any incorrectly generated Ayah or Hadith, ensuring fidelity to the original sources. Evaluation will be based on Word Error Rate (WER). Note that complete Ayahs and complete Hadiths are expected. Writing and diacritics of Ayahs and Hadiths should be obtained from the shared Quran and Hadith sources.

Subtask 1C:
Provide the correction for the claimed Hadith, which should be "مفاتح الغيب خمس لا يعلمها إلا الله", and there is no clear correction for the claimed Ayah2.

Subtask 1C: 

Accuracy. Number of accurate corrections divided by number of total corrections.


Subtask 1C: Correction of Erroneous Content
Task Overview
Subtask 1C: Correction of Erroneous Content. Participants will provide corrected versions for any incorrectly generated Ayah or Hadith, ensuring fidelity to the original sources. Evaluation will be based on Word Error Rate. Note that complete verse(s) from Quran and complete Hadiths are expected. Writing and diacritics of Quran verses and Hadiths should be obtained from the shared Quran and Hadith sources.

How to participate
You agree to use this benchmark for research purposes only. To participate in the shared task, please fill out this form.

Submission Format
Your submission should be a zip file containing a TSV file consisting of two tab-separated columns (Sequence_ID and Correction). Where Correction is the correct intended Ayah/Hadith if one exists or 'خطأ' if there are no similar Ayah/Hadith.

For example, if the Question_ID is C-Q001 and the LLM output is as follows:

'قال الله: "قل هو الله أحد" وأيضا {والله رحيم بالعباد} وقال الرسول 'إنما الأعمال بالنيات

Note that the intended Ayah (والله رحيم بالعباد) is incorrect. The corresponding input file would look like this:

Sequence_ID	Question_ID	Span_Type	Span_Start	Span_End
1	C-Q001	Ayah	34	52

The expected output of the Ayah/Hadith correction program would be

Sequence_ID	Correction
1	وَاللَّهُ رَءُوفٌ بِالْعِبَادِ

If there is no similar Ayah/Hadith, the output should look like this:

Sequence_ID	Correction
2	خطأ
Shared Task Details
See the "Timeline" page for additional information about the phases of this competition.

The website for the IslamicEval 2025 Shared Task is accessible here.



source_files:
datasets/quranic_verses.json
datasets/six_hadith_books.json




Submission Test Data Input:
datasets/1_c_test_data/Test_Subtask_1C_USER.tsv
datasets/1_c_test_data/Test_Subtask_1C.jsonl or datasets/1_c_test_data/Test_Subtask_1C.xml
[Any submissions will need to evaluate using the above files]


Development Set (there is groundtruth):
Input:
datasets/TaskC_GT.tsv -> spans are actually input ->
Columns: Question_ID	Label	Span_Start	Span_End	Original_Span are input - but only 	Correction is ground truth
Also - datasets/TaskC_Input.xml is input (Response part specifically).
[These are to help assessing our code]