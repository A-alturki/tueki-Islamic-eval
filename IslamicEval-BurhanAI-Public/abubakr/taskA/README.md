# PreRequisites

We need to build the **Finetuned Model** for detecting phrases in Quranic verses and Hadith (subtask A), and then build the **searching index** for subtasks B and C.

You need to update `../.env` file with your OpenAI API key.

You need to install the required Python packages. 
You can use the `../requirements.txt` file provided in this directory.`

```bash
pip install -r ../requirements.txt
```

## 1) Finetuning

To finetune a model for detecting phrases in Quranic verses and Hadith, you can follow these steps:

[Read More](finetuning/README.md)

1- Build a synthetic dataset for training.

```bash
cd finetuning
python 01-phrase_detection_dataset_llm_generation.py
```

2- Combine the synthetic dataset with the competition dev dataset to create the final training dataset.

```bash
cd finetuning
python 02-phrase_detection_dataset_creation.py
```

3- Run the finetuning script to train the model.

```bash
cd finetuning
python 03-phrase_detection_finetuning_task.py
```


## 2) Building Islamic Searching Index

Index Both Quran and Hadith for searching

```bash
python 01-index-religion-dataset-for-search.py
```

### Testing the Searching Index

You can run the `02-search-religion-text.py` script to search for Quranic verses or Hadith based on a given phrase. This script will utilize the index created in the previous step.

Update the script to test a query phrase.

```bash
python 02-search-religion-text.py
```

# Building Submissions Files

## Task 1 - A

```bash
python 03-build-subtaskA-submission.py ft:gpt-4.1-mini-2025-04-14:personal:phrase-detection:C1WoOLMK   --test_file ../datasets/taskA-testdata/Test_Subtask_1A/test_Subtask_1A.jsonl --output ./tasks_output/test_Subtask_1A_submission-v1.tsv --verbose
```


## Task 1 - B 

```bash
# Run the script to build the submission for Task B
python 04-build-subtaskB-submission.py --phrases-detected-tsv "../datasets/taskA-testdata/Test_Subtask_1B/Test_Subtask_1B_USER.tsv" --test-data-file "../datasets/taskA-testdata/Test_Subtask_1B/Test_Subtask_1B.jsonl" --output_file_name "tasks_output/submission_task_B_verbose.tsv" --verbose
```

## Task 1 - C
```bash
# Run the script to build the submission for Task C
python 05-build-subtaskC-submission.py --phrases-detected-tsv "../datasets/taskA-testdata/Test_Subtask_1C/Test_Subtask_1C_USER.tsv" --test-data-file "../datasets/taskA-testdata/Test_Subtask_1C/Test_Subtask_1C.jsonl" --output_file_name "tasks_output/submission_task_C.tsv" --verbose
```

---------

- Do not forget to remove **verbose columns** before submission.
- Do not forget to remove the **columns names** from first line in the output files before submission.

