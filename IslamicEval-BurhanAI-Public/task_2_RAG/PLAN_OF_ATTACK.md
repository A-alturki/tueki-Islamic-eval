TASK 2:

Documentation: 
original_docs/task_2_docs_and_organizers_code

* Given that we don't have a training phase, we will need to merge training and dev data into one set as well as the gold ground-truth, so that we can use them to understand the performance of the system.


* Next - implement a pipeline to submit the combined set in parallel to the large language model - and we must do post processing to basically get the format they need for the submission. For example, the output from the LLM will give individual Ayas and Hadeeths - for Hadeeth it is ok - we need to cite the number of the hadeeth (maybe we can do a sanity check to see if the text in the JSON response from the LLM actually belongs to the hadeeth number using the Hadeeth database we have in datasets/task_2_data/Sahih-Bukhari/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl) and report any issues for analysis.

But for Quran - the response from the LLM will give individual Ayas - which means, I have to look at the file datasets/task_2_data/Thematic_QPC/QH-QA-25_Subtask2_QPC_v1.1.tsv - and see which number I have to return (i.e. the LLM returned Sura 1 Aya 1 and next item is Sura 1 Aya 3, and in the file I mentioned - there is a line saying 1:1-4 - which means Sura 1 and Aya 1 to 4 - that means - the two items from the LLM belong to one item in the output). I also need to highlight issues related to the duplicates or low quality results for later analysis in the processing logs.

 and then evaluate and analyse the results and reasons of failures thoroughly, and allow choice of different models and with different reasoning effort - set verbosity to medium always. Use original_docs/task_2_docs_and_organizers_code/Evaluation code to do the evaluation - please don't make any changes to this code and invoke as is.

* Next - we need to ensure that record each run with the right name inside outputs_task_2/burhanai_task2_RAG_{{modelname}}_{{effort}}_{{any other qualifires}}_{{timestamp}} - this applies to folder name and also file name that we will submit. Ensure that you call the format checker to ensure that our submission is correct. Use original_docs/task_2_docs_and_organizers_code/Evaluation/checker.py for this task - don't make any changes to it please.


Make sure that you process things in a way that allows for continuation if things cut off mid way, also retry and so on - same as in task c and a code base, also progress repoting is critical (same as c and a) and also - ensure that the code is well written, clear, easy to understand and efficient.





Code for the LLM calling:
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
          "text": "Give me UP TO 20 (maybe less or even none) (ordered) items (from quran or Saheeh Albukhari only - not the full Sunnah) that are potentially enclose the answer(s) of the question below (ideally fully but maybe partially):\nمن الذي خسف الله به الأرض؟\n\n\nRules:\nThe results MUST potentially enclose the answer(s) to the given question (don't return things that are vaguely relevant or related but don't answer the question).\nYOU MUST MAKE AS MANY SEARCHES INTO THE FILE STORE ATTACH AS YOU NEED - YOU MUST ONLY RETURN VALID RESULTS STRICTLY IN THE FORMAT AND USE THE FILES ATTACHED - WHICH CONTAIN WHOLE QURAN AND SAHEEH ALBUKHARI\nYou must rank the results based on how much they answer the question (i.e. citations that answer the question fully and directly are first and less so later in the list) - this is critical.\n\nGive me the following format for each result:\nJSON - list of citations, each citation has type (e.g. quran or hadeeth strict enum), for hadeeth, you must give hadeeth text and number (strictly from the file in the file search), and then, if quran, you must provide aya text, and then sura number (as integer) and aya number (integer).  If It is Hadeeth then keep Quran fields as \"N/A\" and vice versea\n\nIf there are less than 20 that are relevant it is ok - return less - only the ones that strictly follow the task are allowed.\n\nAlso - to allow you to do the right search - guess which texts might be relevant from your memory and use them to inform your search and also use the question text to formulate multiple searches that could lead to the right texts - do as many searches you need.\n\nIf there is no citations that follow the rules above and don't enclose the answer to the questions ideally fully or even partially directly  - you can return an empty answer. - but this is less likely."
        }
      ]
    },
    {
      "type": "reasoning",
      "id": "rs_689775f396b081a3a69734053b132ba70d02c9e897daa16a",
      "summary": [
        {
          "type": "summary_text",
          "text": "**Reconciling citation requirements**\n\nI'm thinking about how to handle citation markers. The tool suggests including them in my responses, ideally right after the aya_text or hadeeth_text. However, doing that would change the original text, which may not be a good idea. An alternative could be placing citation markers outside the JSON, but the developer's strict format requires a JSON list of citations. Ugh! It feels like a bit of a conflict since I must follow the system's requirements."
        },
        {
          "type": "summary_text",
          "text": "**Evaluating citation guidelines**\n\nI'm grappling with the developer's rules versus the system's requirements. The system message indicates that including markers is a higher priority. If I append markers to \"aya_text\" and \"hadeeth_text,\" it could alter the original text, which isn't allowed since the developer requires exact wording from the source. For the Quran, the same exactness applies. I'm considering alternatives, like adding a \"source\" property outside the text fields. However, the schema is strict, so I need to find a suitable solution that fits within the guidelines."
        },
        {
          "type": "summary_text",
          "text": "**Clarifying citation requirements**\n\nI'm looking at the citation format requirements, which specify a JSON list where each citation has a type, and for the Quran, it includes sura and aya numbers. For Hadeeth, those fields must be \"N/A.\" I know there can't be extra fields, but maybe this isn't strictly enforced. I wonder if the Quran and Bukhari can be identified by just those numbers, eliminating the need for a \"filecite\" marker. However, the tool still insists on including these markers. It’s a bit confusing! What's the priority here?"
        },
        {
          "type": "summary_text",
          "text": "**Considering citation format**\n\nI'm working on how to incorporate citation markers into my responses. The tool suggests including these markers, but the developer's format might limit that. The citation details, like sura and hadith numbers, could serve as identifiers. However, since the tool emphasizes including markers in the responses, I’m wondering if I could introduce a \"file_citation\" field to hold these markers without violating the JSON schema. They didn’t outright say not to add extra fields; it’s a bit tricky!"
        },
        {
          "type": "summary_text",
          "text": "**Deciding on citation format**\n\nThe instructions stress including specific fields in a JSON list for each citation, particularly for Quran and Hadeeth references. It seems safest to stick to those required fields and not risk adding extra ones. While the tool demands citation markers, maybe they aren't strictly enforced for structured outputs. It’s tempting to include a \"file_citation\" field but that still feels risky since the schema doesn't allow it. I guess I'll omit the markers and hope that’s sufficient, then proceed with my search queries."
        }
      ]
    },
    {
      "id": "fs_6897762a3ffc81a3bed53f770975c4640d02c9e897daa16a",
      "type": "file_search_call",
      "status": "completed",
      "queries": [
        "من الذي خسف الله به الأرض؟",
        "فخسفنا به وبداره الأرض",
        "قارون فخسفنا به الأرض سورة القصص 28:81",
        "ومنهم من خسفنا به الأرض سورة العنكبوت 29:40",
        "يخسف بالجيش البيداء صحيح البخاري"
      ],
      "results": null
    },
    {
      "type": "reasoning",
      "id": "rs_6897762a654c81a3a6a92221b17ee3d40d02c9e897daa16a",
      "summary": []
    },
    {
      "id": "fs_68977630635c81a3907b2f40697f96140d02c9e897daa16a",
      "type": "file_search_call",
      "status": "completed",
      "queries": [
        "يُخْسَفُ بِأَوَّلِهِمْ وَآخِرِهِمْ البيداء صحيح البخاري",
        "يخسف بالجيش البيداء يخسف بأولهم وآخرهم s_buk.txt",
        "يغزو جيش الكعبة فإذا كانوا ببيداء من الأرض يخسف بأولهم وآخرهم صحيح البخاري رقم الحديث"
      ],
      "results": null
    },
    {
      "type": "reasoning",
      "id": "rs_68977633c20c81a39b1239a6cf9babd20d02c9e897daa16a",
      "summary": [
        {
          "type": "summary_text",
          "text": "**Citing sources on Qarun**\n\nThe question specifically uses \"الذي,\" indicating a singular reference, but including hadith might not directly answer \"who was the one.\" I think the best approach is to provide citations that clearly identify Qarun as the one swallowed by the earth using Surah al-Qasas 28:76 and 28:81. Following the developer's note to limit to \"up to 20,\" I feel it's safe to return just those two. Other citations may be optional, but including 29:39 and 29:40 can provide extra context."
        },
        {
          "type": "summary_text",
          "text": "**Compiling citations for Qarun**\n\nI think including 29:39-40 seems relevant since they provide context. I'll add these two verses and also include 28:79 to confirm Qarun's name. While 28:81 is critical because of \"خسف,\" 28:79 remains relevant for identification. The question asks, \"Who did Allah cause the earth to swallow?\" The answer is Qarun, which 28:81 shows. Now, let’s compile a list of relevant citations including the verses in order, as well as a hadith about \"يغزو جيش الكعبة.\""
        },
        {
          "type": "summary_text",
          "text": "**Assessing citation relevance**\n\nThe question uses \"الذي,\" which indicates a singular subject; perhaps it's meant as a puzzle. Including hadith might be overkill since the developer only required Quran or Bukhari. I think hadith should rank lower in the list since they aren't directly essential. They asked for \"up to 20\" items that potentially enclose answers. While the hadith about the army tangentially relates to swallowing, it doesn't directly answer \"who.\" If the answer is \"Qarun,\" then it's likely unnecessary to include the hadith."
        },
        {
          "type": "summary_text",
          "text": "**Identifying relevant citations for Qarun**\n\nTo ensure clarity, I want to include only citations that help directly identify Qarun. Including the hadith as an example of \"خسف بالأرض\" might be tangential and doesn't specifically answer \"who was 'the one'.\" The tool advises against adding vague references, which suggests the hadith about a future army shouldn’t be in the list. Thus, my final citations should focus on:\n\n- S. Al-Qasas 28:81: directly describes the event.\n- S. Al-Qasas 28:76: explicitly names Qarun.\n- Optionally, 28:79 provides additional context, but 28:76 suffices.\n- S. Al-Qasas 28:82: offers context but isn't necessary."
        },
        {
          "type": "summary_text",
          "text": "**Ranking citations for clarity**\n\nI'm looking at how to rank citations that help identify Qarun while also including context. The first verse, 28:81, clearly details the event of him being swallowed by the earth. The second verse, 28:76, explicitly names Qarun, making it essential. I want to include 29:40 and 29:39 for context, even if 29:40 doesn't directly identify who it refers to. Finally, I’ll add 28:79 and 28:82 for additional context, though they are less crucial. The ranking then would be as follows:\n\n1. 28:81.\n2. 28:76.\n3. 28:79.\n4. 29:39.\n5. 29:40.\n6. 28:82."
        }
      ]
    },
    {
      "id": "fs_6897767669bc81a396f0939c2de8f2220d02c9e897daa16a",
      "type": "file_search_call",
      "status": "completed",
      "queries": [
        "بينما رجل يمشي في برديه خسف به الأرض صحيح البخاري",
        "رجل يجر إزاره خسف به الأرض صحيح البخاري",
        "انخسفت به الأرض يجر إزاره البخاري"
      ],
      "results": null
    },
    {
      "type": "reasoning",
      "id": "rs_6897767a343881a3a6e4d4d59da47ebb0d02c9e897daa16a",
      "summary": []
    },
    {
      "id": "fs_6897767eb0d481a3947e104df74af53c0d02c9e897daa16a",
      "type": "file_search_call",
      "status": "completed",
      "queries": [
        "يتجلجل في الأرض إلى يوم القيامة صحيح البخاري خسف به",
        "بينما رجل يمشي في حلة تعجبه نفسه مرجل جمته خسف به فهو يتجلجل صحيح البخاري",
        "خسف به فهو يتجلجل في الأرض إلى يوم القيامة البخاري"
      ],
      "results": null
    },
    {
      "type": "reasoning",
      "id": "rs_6897768203b081a38fc90b9e564d9a620d02c9e897daa16a",
      "summary": []
    },
    {
      "id": "msg_689776937b5481a38f8f2276633c7a590d02c9e897daa16a",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "{\n  \"citations\": [\n    {\n      \"type\": \"quran\",\n      \"hadeeth_text\": \"N/A\",\n      \"hadeeth_number\": \"N/A\",\n      \"aya_text\": \"فَخَسَفْنَا بِهِ وَبِدَارِهِ الْأَرْضَ فَمَا كَانَ لَهُ مِنْ فِئَةٍ يَنْصُرُونَهُ مِنْ دُونِ اللَّهِ وَمَا كَانَ مِنَ الْمُنْتَصِرِينَ\",\n      \"sura_number\": 28,\n      \"aya_number\": 81\n    },\n    {\n      \"type\": \"quran\",\n      \"hadeeth_text\": \"N/A\",\n      \"hadeeth_number\": \"N/A\",\n      \"aya_text\": \"۞ إِنَّ قَارُونَ كَانَ مِنْ قَوْمِ مُوسَىٰ فَبَغَىٰ عَلَيْهِمْ ۖ وَآتَيْنَاهُ مِنَ الْكُنُوزِ مَا إِنَّ مَفَاتِحَهُ لَتَنُوءُ بِالْعُصْبَةِ أُولِي الْقُوَّةِ إِذْ قَالَ لَهُ قَوْمُهُ لَا تَفْرَحْ ۖ إِنَّ اللَّهَ لَا يُحِبُّ الْفَرِحِينَ\",\n      \"sura_number\": 28,\n      \"aya_number\": 76\n    },\n    {\n      \"type\": \"quran\",\n      \"hadeeth_text\": \"N/A\",\n      \"hadeeth_number\": \"N/A\",\n      \"aya_text\": \"فَخَرَجَ عَلَىٰ قَوْمِهِ فِي زِينَتِهِ ۖ قَالَ الَّذِينَ يُرِيدُونَ الْحَيَاةَ الدُّنْيَا يَا لَيْتَ لَنَا مِثْلَ مَا أُوتِيَ قَارُونُ إِنَّهُ لَذُو حَظٍّ عَظِيمٍ\",\n      \"sura_number\": 28,\n      \"aya_number\": 79\n    },\n    {\n      \"type\": \"quran\",\n      \"hadeeth_text\": \"N/A\",\n      \"hadeeth_number\": \"N/A\",\n      \"aya_text\": \"وَقَارُونَ وَفِرْعَوْنَ وَهَامَانَ ۖ وَلَقَدْ جَاءَهُمْ مُوسَىٰ بِالْبَيِّنَاتِ فَاسْتَكْبَرُوا فِي الْأَرْضِ وَمَا كَانُوا سَابِقِينَ\",\n      \"sura_number\": 29,\n      \"aya_number\": 39\n    },\n    {\n      \"type\": \"quran\",\n      \"hadeeth_text\": \"N/A\",\n      \"hadeeth_number\": \"N/A\",\n      \"aya_text\": \"فَكُلًّا أَخَذْنَا بِذَنْبِهِ ۖ فَمِنْهُمْ مَنْ أَرْسَلْنَا عَلَيْهِ حَاصِبًا وَمِنْهُمْ مَنْ أَخَذَتْهُ الصَّيْحَةُ وَمِنْهُمْ مَنْ خَسَفْنَا بِهِ الْأَرْضَ وَمِنْهُمْ مَنْ أَغْرَقْنَا ۚ وَمَا كَانَ اللَّهُ لِيَظْلِمَهُمْ وَلَـٰكِنْ كَانُوا أَنْفُسَهُمْ يَظْلِمُونَ\",\n      \"sura_number\": 29,\n      \"aya_number\": 40\n    },\n    {\n      \"type\": \"quran\",\n      \"hadeeth_text\": \"N/A\",\n      \"hadeeth_number\": \"N/A\",\n      \"aya_text\": \"وَأَصْبَحَ الَّذِينَ تَمَنَّوْا مَكَانَهُ بِالْأَمْسِ يَقُولُونَ وَيْكَأَنَّ اللَّهَ يَبْسُطُ الرِّزْقَ لِمَنْ يَشَاءُ مِنْ عِبَادِهِ وَيَقْدِرُ ۖ لَوْلَا أَنْ مَنَّ اللَّهُ عَلَيْنَا لَخَسَفَ بِنَا ۖ وَيْكَأَنَّهُ لَا يُفْلِحُ الْكَافِرُونَ\",\n      \"sura_number\": 28,\n      \"aya_number\": 82\n    }\n  ]\n}"
        }
      ]
    }
  ],
  text={
    "format": {
      "type": "json_schema",
      "name": "quran_and_bukhari_citations",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "citations": {
            "type": "array",
            "description": "An ordered list (up to 20, possibly less or none) of citations from the Quran or Saheeh Bukhari that may potentially enclose the answer(s) to the user's question. The list is strictly ranked according to directness and completeness of the answer.",
            "items": {
              "type": "object",
              "properties": {
                "type": {
                  "type": "string",
                  "description": "Source of the citation, either 'quran' or 'hadeeth'.",
                  "enum": [
                    "quran",
                    "hadeeth"
                  ]
                },
                "hadeeth_text": {
                  "type": "string",
                  "description": "Exact hadeeth text from Saheeh Bukhari as appears in the source file included in the file search tool if type is 'hadeeth'; otherwise 'N/A'."
                },
                "hadeeth_number": {
                  "type": "string",
                  "description": "ID or number of the hadeeth as strictly found in Saheeh Bukhari source included in the file search tool, if type is 'hadeeth'; otherwise 'N/A'."
                },
                "aya_text": {
                  "type": "string",
                  "description": "Ayah text from the Quran if type is 'quran'; otherwise 'N/A'."
                },
                "sura_number": {
                  "type": "integer",
                  "description": "Sura number (1-114) if type is 'quran'; otherwise -1."
                },
                "aya_number": {
                  "type": "integer",
                  "description": "Ayah number within the sura if type is 'quran'; otherwise -1."
                }
              },
              "required": [
                "type",
                "hadeeth_text",
                "hadeeth_number",
                "aya_text",
                "sura_number",
                "aya_number"
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
    "verbosity": "medium"
  },
  reasoning={
    "effort": "high",
    "summary": "detailed"
  },
  tools=[
    {
      "type": "file_search",
      "vector_store_ids": [
        "vs_68976251213c8191b42764ef86684a83"
      ]
    }
  ],
  store=False
)