## Installation

Before executing any of the given scripts, perform the following steps:

1. Clone the repository
    ~~~
    $ git clone https://gitlab.com/bigirqu/quran-hadith-qa-2025.git
    ~~~

2. Create a virtual environment for the project using venv or conda and activate it.


3. pip install the packages in the ['requirements.txt'].

    ```
    $ pip install -r  Evaluation/requirements.txt
    ```


## Submission checker script

It is highly recommended to use this script to verify your ***run file*** (prior to submission) with respect to the file name, the correctness of its format, and checking for duplicates, etc. 

The expected run file is in TREC run format with **tsv** as a file extension. Check the task website [here](https://sites.google.com/view/islamiceval-2025/subtask-2/subtask-2-run-submission) for full detail about the run format and file naming.


Here is an example of executing the submission checker script:

```
python ./Evaluation/checker.py -m "./Evaluation/input/res/run_sample.tsv" # change this to the path of your run file
```


## Evaluation script

To evaluate your run file, go to "Evaluation/evaluate.py" and update the input paths to your run file and to the dev/train QRELs files. The current script is set to run on a sample run and qrels. You can run the evaluation script as follows:

```
python Evaluation/evaluate.py
```
