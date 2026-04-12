import json
import sys
import os

sys.path.append(os.path.abspath("."))
# Load results
with open('results/eval_nfcorpus.json') as f:
    data = json.load(f)

# Load qrels
from evaluation.dataset_loader import DatasetLoader

loader = DatasetLoader('data/nfcorpus')
qrels = loader.load_qrels()

# 🔍 Debug prints
print("Sample RESULT query_id:", list(data.keys())[0])

first_qid = list(qrels.keys())[0]
print("Sample QREL query_id:", first_qid)

print("Sample QREL doc_id:", list(qrels[first_qid].keys())[0])

print("Total QREL queries:", len(qrels))
print("Total RESULT queries:", len(data))

# 🔥 Check overlap
common = set(data.keys()) & set(qrels.keys())
print("Common query IDs:", len(common))