import pandas as pd
import os, json
import checker
from trectools import TrecEval, TrecRun, TrecQrel




def evaluate(input_run, qrels, depth):
    trec_run = TrecRun(input_run)
    trec_eval = TrecEval(trec_run, qrels)
    res = round(trec_eval.get_map(depth=depth), 4)
    return res

    
def add_unscored_questions(df, qids):

    existing_ids = set(df['query'])
    missing_ids = [i for i in qids if i not in existing_ids]
    # Create new rows for the unretrieved questions with -1 as docid
    new_rows = pd.DataFrame({
        'query': missing_ids,
        "q0": "Q0",
        'docid': -1,
        'rank': 1,
        'score': 10,
        'system': "unretrieved_question"
    })
    df = pd.concat([df, new_rows], ignore_index=True)
    return df




current_dir = os.path.dirname(os.path.abspath(__file__))
reference_dir = os.path.join(current_dir, 'input/', 'ref')
# set the path to your run file here
prediction_dir = os.path.join(current_dir, 'input/', 'res')
run_file =  os.path.join(prediction_dir, "run_sample.tsv")

score_dir = os.path.join(current_dir, 'output/')
hadith_run_file = os.path.join(prediction_dir, "hadith_run.tsv" )
quran_run_file = os.path.join(prediction_dir, "quran_run.tsv")

hadith_qrels =  os.path.join(reference_dir, "hadith_sample.qrels")
quran_qrels =  os.path.join(reference_dir, "quran_sample.qrels")
combined_qrels_file =  os.path.join(reference_dir, "combined_qrels.tsv")

format_check_passed = checker.check_run(run_file)
if not format_check_passed:
    print("script stopped due to incorrect run format")
    exit(0)


h_qrels = TrecQrel(hadith_qrels)
q_qrels = TrecQrel(quran_qrels)
combined_qrels = TrecQrel(combined_qrels_file)

# unique ids of all questions
qids = list(set(combined_qrels.qrels_data['query']))

input_run = TrecRun(run_file)
quran_run = TrecRun()
hadith_run = TrecRun()
rundata = input_run.run_data

# separate the run into two: hadith and quran runs
# the distinction is that quran passages contain : in their ids
quran_run.run_data = rundata[rundata['docid'].str.contains(':', na=False)]
hadith_run.run_data = rundata[~rundata['docid'].str.contains(':', na=False)]


# add -1 for the unscored questions in quran and hadith runs
quran_run.run_data = add_unscored_questions(quran_run.run_data, qids)
hadith_run.run_data = add_unscored_questions(hadith_run.run_data, qids)


hadith_run.run_data.to_csv(hadith_run_file, index=False, header=False, sep='\t')
quran_run.run_data.to_csv(quran_run_file, index=False, header=False, sep='\t')


r1_comb_map = evaluate(run_file, combined_qrels, depth=10)
r1_quran_map = evaluate(quran_run_file, q_qrels, depth=5)
r1_hadith_map = evaluate(hadith_run_file, h_qrels, depth=5)

print(f"MAP@10 = {r1_comb_map}")
print(f"MAP_Q@5 = {r1_quran_map}")
print(f"MAP_H@5 = {r1_hadith_map}")

scores = {
    'MAP@10': r1_comb_map,
    'MAP_Q@5': r1_quran_map,
    'MAP_H@5': r1_hadith_map
}

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))