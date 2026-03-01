# Qur'an Hadith QA 2025

This repository contains the datasets, format checker and scorer for the [Qur&#39;an and Hadith QA 2025 Subtask 2 of IslamicEval 2025 shared task](https://sites.google.com/view/islamiceval-2025)

This subtask is a continuation of [Qur'an QA 2022](https://sites.google.com/view/quran-qa-2022) and [Qur'an QA 2023](https://sites.google.com/view/quran-qa-2023) shared tasks.

The subtask this year is defined as follows: Given a free-text question posed in MSA, a collection of Qur'anic passages (that cover the Holy Qur'an) and a collection of Hadiths from Sahih Al-Bukhari, a system is required to retrieve a ranked list of up-to 20 answer-bearing Qur'anic passages or Hadiths (i.e., Islamic sources that potentially enclose the answer(s) to the given question) from the two collections. The question can be a factoid or non-factoid question. 


To make the task more realistic (thus challenging), some questions may not have an answer in the Holy Qur'an and Sahih Al-Bukhari. In such cases, the ideal system should return no answers; otherwise, it returns a ranked list of up to 20 answer-bearing sources.


The datasets for Subtask 2 are composed of the following collections and datasets.
* The Qur'anic Passage collection (QPC).
* The Sahih Al-Bukhari collection.
* The questions of the *AyaTEC* dataset and their relevance judgments over the Qur'anic Passage collection **only**.

Since the Hadith QA component of Subtask 2 is newly introduced in this year's shared task, participating teams are encouraged to utilize any existing Hadith QA resources for training their models and systems.

## [Thematic Qur&#39;an Passage Collection (QPC)](https://gitlab.com/bigirqu/quran-hadith-qa-2025/-/tree/main/data/Thematic_QPC)

This file contains 1,266 thematic Qur'anic passages that cover the whole Holy Qur'an. Thematic passage segmentation was conducted using the Thematic Holy Qur'an [1] https://surahquran.com/tafseel-quran.html. This tsv file has the following format:

    `<passage-id>` `<passage-text>`

where the passage-id has the format: *Chapter#:StartVerse#-EndVerse#*, and the passage-text (i.e., Qur’anic text) was taken from the normalized simple-clean text style (from Tanzil 1.0.2) https://tanzil.net/download/.

## [Sahih Al-Bukhari Hadith Collection](https://gitlab.com/bigirqu/quran-hadith-qa-2025/-/tree/main/data/Sahih-Bukhari)

This collection comprises 2,254 Hadiths, from which all redundant Hadiths and Arabic commentary have been excluded by the authors [2]. This collection is in JSONL format. <!-- has the following format:-->


## [AyaTEC_v1.3 Dataset](https://gitlab.com/bigirqu/quran-hadith-qa-2025/-/tree/main/data)

AyaTEC_v1.3 is composed of 300 questions, 250 of which are taken from the [AyaTEC_v1.2 dataset](https://gitlab.com/bigirqu/quran-qa-2023/-/tree/main/Task-A/data), and 50 test questions to be released on July 20.

<!--, in addition to 52 new test questions for evaluating the systems in the PR task (Task A). --->
So far, AyaTEC includes a total of 37 *zero-answer* questions that do not have an answer in the Holy Qur’an. The distribution of the training, development, and test splits are shown below.
<!-- The differences between  the AyaTEC_v1.2 and AyaTEC_v1.1 datasets are listed [here](https://gitlab.com/bigirqu/quran-qa-2023/-/blob/main/Task-A/data/README.md).-->

| **Dataset** | **Questions** | **Question-Passage Pairs*** |
| ----------------- | :-------------------: | :--------------------------------: |
| Training          |          210          |                1261               |
| Development       |          40          |                298                |
| Test              |           50          |              TBA                  |

<!--
| **Dataset** | **%** | **# Questions** | **Question-Passage Pairs***|
|-------------|:-----:|:---------------:|:-------------------------:|
| Training    |  70%  |       174       |            972            |
| Development |  10%  |        25       |            160            |
| Test        |  20%  |        52       |            427            |
| All         | 100%  |       251       |          1,599            |--->

*Question-Passage pairs are included in the QRels datasets.


 These datsets are tab-delimted with the following format:

    `<question-id>`  `<question-text>`

The text encoding in all datasets is UTF-8.

## [The QRels Gold Datasets](https://gitlab.com/bigirqu/quran-hadith-qa-2025/-/tree/main/data/qrels)

The query relevance judgements (QRels) datasets are jointly composed of 1,559  gold (answer-bearing) Qur'anic passage-IDs considered relevant to each question. For *zero-answer* questions, the `passage-id` will a have a value of "-1". The distribution of the QRels are shown in the table above, and they adopt the following [TREC format](https://trec.nist.gov/data/qrels_eng/):

    `<question-id>` Q0 `<passage-id>` `<relevance>`

<!--## Evaluation

The following scripts are needed for evaluating the submissions to Task A:

* A [*submission checker* script](https://gitlab.com/bigirqu/quran-qa-2023/-/tree/main/Task-A/code) for checking the correctness of run files to be submitted.
* An [*evaluation* (or *scorer*) script](https://gitlab.com/bigirqu/quran-qa-2023/-/tree/main/Task-A/code).-->

## References

[1] Swar, M. N., 2007. Mushaf Al-Tafseel Al-Mawdoo’ee. Damascus: Dar Al-Fajr Al-Islami.

[2] Al-Sharjy, A. B. A and Al-Zubaidi, Z., Al-Tajreed Al-Sareeh of  Collective Sahih Hadith, التجريد الصريح لأحاديث الجامع الصحيح, 2009.



