# evaluation/dataset_loader.py

import json
import csv
import os


class DatasetLoader:
    """
    Loads BEIR-format datasets (SciFact, NFCorpus, etc.)

    BEIR format:
        corpus.jsonl  — {_id, title, text}
        queries.jsonl — {_id, text}
        qrels/*.tsv   — query_id, doc_id, relevance_score

    Relevance scales:
        SciFact  — binary (0 or 1)
        NFCorpus — graded (0, 1, 2, 3)  → we keep anything >= 1
    """

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.corpus_path  = os.path.join(dataset_path, "corpus.jsonl")
        self.queries_path = os.path.join(dataset_path, "queries.jsonl")

        # qrels path — try test.tsv first, fallback to dev.tsv
        # NFCorpus ships with dev.tsv instead of test.tsv
        test_path = os.path.join(dataset_path, "qrels", "test.tsv")
        dev_path  = os.path.join(dataset_path, "qrels", "dev.tsv")

        if os.path.exists(test_path):
            self.qrels_path = test_path
        elif os.path.exists(dev_path):
            self.qrels_path = dev_path
            print(f"[INFO] test.tsv not found, using dev.tsv for qrels")
        else:
            raise FileNotFoundError(
                f"No qrels file found in {os.path.join(dataset_path, 'qrels')} — "
                f"expected test.tsv or dev.tsv"
            )

    def load_corpus(self) -> dict:
        """
        Load all documents from corpus.jsonl.

        Returns:
            dict — {doc_id: {"title": str, "text": str}}
        """
        corpus = {}
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc    = json.loads(line.strip())
                doc_id = str(doc["_id"])
                corpus[doc_id] = {
                    "title": doc.get("title", ""),
                    "text":  doc.get("text",  ""),
                }
        print(f"Loaded {len(corpus)} documents from corpus")
        return corpus

    def load_queries(self) -> dict:
        """
        Load test queries from queries.jsonl.

        Returns:
            dict — {query_id: query_text}
        """
        queries = {}
        with open(self.queries_path, "r", encoding="utf-8") as f:
            for line in f:
                q = json.loads(line.strip())
                queries[str(q["_id"])] = q["text"]
        print(f"Loaded {len(queries)} queries")
        return queries

    def load_qrels(self) -> dict:
        """
        Load relevance judgments from qrels file.

        Handles both:
            SciFact  — binary relevance (0 or 1)
            NFCorpus — graded relevance (0, 1, 2, 3) → keep score >= 1

        Returns:
            dict — {query_id: {doc_id: relevance_score}}
        """
        qrels = {}

        with open(self.qrels_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header: query-id  corpus-id  score

            for row in reader:
                if len(row) < 3:
                    continue

                query_id = str(row[0])
                doc_id   = str(row[1])
                score    = int(row[2])

                # skip completely irrelevant docs
                # this handles both binary (0/1) and graded (0/1/2/3)
                if score < 1:
                    continue

                if query_id not in qrels:
                    qrels[query_id] = {}

                qrels[query_id][doc_id] = score

        print(f"Loaded qrels for {len(qrels)} queries "
              f"from {os.path.basename(self.qrels_path)}")
        return qrels


if __name__ == "__main__":
    import sys

    # pass dataset path as argument or default to scifact
    # usage: python -m evaluation.dataset_loader data/nfcorpus
    path   = sys.argv[1] if len(sys.argv) > 1 else "data/scifact"
    loader = DatasetLoader(path)

    corpus  = loader.load_corpus()
    queries = loader.load_queries()
    qrels   = loader.load_qrels()

    # show a sample
    sample_qid = list(queries.keys())[0]
    print(f"\nSample query  [{sample_qid}]: {queries[sample_qid]}")
    print(f"Relevant docs : {qrels.get(sample_qid, {})}")