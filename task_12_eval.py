import json
import sys

#Call as: python task_12_eval.py <dataset_file> <output_file_name>
#example: python task_12_eval.py questions.jsonl predictions.json

#Read in gold truth into a dictionary
def get_golds(questions):

    golds = {}
    for question_line in questions:
        question = json.loads(question_line)
        uuid = question['uuid']
        gold_answers = question['golden_answer'].split() #Can be multipl "B,C"
        #gold_answers is a list of all correct answers
        assert uuid not in golds, "There should not be duplicated uuids"
        golds[uuid]= gold_answers
    return golds

def score(preds, golds):
    assert len(preds) == len(golds), "There is a mismatch in number of instances"
    points = 0.0
    for curr_id in golds:
        curr_gold = golds[curr_id]
        curr_pred = preds[curr_id]
        if(curr_gold == curr_pred):
            points += 1.0 #complete match
        elif any(i in curr_gold for i in curr_pred):
            points += 0.5 #Partial match

    return points/len(preds)

#Load the questions and docs (json) into dictionaries

# dataset_file = sys.argv[1]
# output_file = sys.argv[2]
dataset_file = "questions.jsonl"
output_file = "out2.json"

questions_lines = open(dataset_file, encoding='utf-8')

golds =  get_golds(questions_lines)
preds = json.load(open(output_file, 'r'))

#Get fraction of total possible points
frac_score = score(preds, golds)
print(frac_score)

