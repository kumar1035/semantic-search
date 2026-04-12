# evaluation/evaluator.py

import math
from collections import defaultdict


class Evaluator:
    """
    Computes standard IR evaluation metrics by comparing your
    system's ranked results against the ground-truth qrels.

    Metrics implemented:
        NDCG@k   — Normalized Discounted Cumulative Gain
                   Measures ranking quality; rewards relevant docs appearing early
                   Handles graded relevance (NFCorpus 0-3) and binary (SciFact 0-1)
        MAP@k    — Mean Average Precision
                   Average of precision computed at each relevant doc position
        Recall@k — Fraction of relevant docs found in top-k
        P@k      — Precision at k (fraction of top-k that are relevant)
        MRR      — Mean Reciprocal Rank (position of first relevant result)
    """

    def ndcg_at_k(self, ranked: list, relevant: dict, k: int) -> float:
        """
        NDCG@k — the most important metric for ranked retrieval.
        Score of 1.0 = perfect ranking, 0.0 = no relevant docs found.

        Works for both:
            - Binary relevance (SciFact): scores are 0 or 1
            - Graded relevance (NFCorpus): scores are 0, 1, 2, or 3
        """
        dcg = 0.0
        for i, (doc_id, _) in enumerate(ranked[:k]):
            rel = relevant.get(doc_id, 0)
            if rel > 0:
                dcg += rel / math.log2(i + 2)   # i+2 because log2(1) = 0

        # Ideal DCG — best possible ranking given the relevant docs
        ideal_rels = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(ideal_rels)
            if rel > 0
        )

        return dcg / idcg if idcg > 0 else 0.0

    def map_at_k(self, ranked: list, relevant: dict, k: int) -> float:
        """
        MAP@k — average precision across all relevant document positions.

        For graded relevance (NFCorpus), any score >= 1 counts as relevant.
        """
        num_relevant  = 0
        sum_precision = 0.0

        for i, (doc_id, _) in enumerate(ranked[:k]):
            if relevant.get(doc_id, 0) > 0:
                num_relevant  += 1
                sum_precision += num_relevant / (i + 1)

        total_relevant = sum(1 for v in relevant.values() if v > 0)
        if total_relevant == 0:
            return 0.0
        return sum_precision / total_relevant

    def recall_at_k(self, ranked: list, relevant: dict, k: int) -> float:
        """
        Recall@k — what fraction of all relevant docs appear in top-k.

        For graded relevance, any score >= 1 counts as relevant.
        """
        total_relevant = sum(1 for v in relevant.values() if v > 0)
        if total_relevant == 0:
            return 0.0
        found = sum(
            1 for doc_id, _ in ranked[:k]
            if relevant.get(doc_id, 0) > 0
        )
        return found / total_relevant

    def precision_at_k(self, ranked: list, relevant: dict, k: int) -> float:
        """
        P@k — fraction of the top-k results that are relevant.

        For graded relevance, any score >= 1 counts as relevant.
        """
        if k == 0:
            return 0.0
        hits = sum(
            1 for doc_id, _ in ranked[:k]
            if relevant.get(doc_id, 0) > 0
        )
        return hits / k

    def mrr(self, ranked: list, relevant: dict) -> float:
        """
        MRR — reciprocal of the rank of the first relevant result.
        Score of 1.0 = first result is relevant.

        For graded relevance, any score >= 1 counts as relevant.
        """
        for i, (doc_id, _) in enumerate(ranked):
            if relevant.get(doc_id, 0) > 0:
                return 1.0 / (i + 1)
        return 0.0

    def evaluate(
        self,
        all_results: dict,
        qrels: dict,
        k_values: list = None,
    ) -> dict:
        """
        Compute all metrics across all queries and average them.

        Args:
            all_results — {query_id: [(doc_id, score), ...]}  from QueryRunner
            qrels       — {query_id: {doc_id: relevance}}     from DatasetLoader
            k_values    — list of k values e.g. [1, 5, 10, 100]

        Returns:
            dict — {
                "NDCG@10": 0.42,
                "MAP@100": 0.38,
                "Recall@100": 0.71,
                "P@10": 0.15,
                "MRR": 0.55,
                "num_queries": 300,
                "queries_with_results": 298,
                "queries_with_no_qrels": 2,
            }
        """
        if k_values is None:
            k_values = [1, 5, 10, 100]

        scores               = defaultdict(list)
        num_queries          = 0
        queries_with_results = 0
        queries_no_qrels     = 0

        for query_id, ranked in all_results.items():
            relevant = qrels.get(query_id, {})

            # skip queries that have no ground truth at all
            if not relevant:
                queries_no_qrels += 1
                continue

            num_queries += 1
            if ranked:
                queries_with_results += 1

            for k in k_values:
                scores[f"NDCG@{k}"].append(self.ndcg_at_k(ranked, relevant, k))
                scores[f"MAP@{k}"].append(self.map_at_k(ranked, relevant, k))
                scores[f"Recall@{k}"].append(self.recall_at_k(ranked, relevant, k))
                scores[f"P@{k}"].append(self.precision_at_k(ranked, relevant, k))

            scores["MRR"].append(self.mrr(ranked, relevant))

        # Print diagnostic so you can see if queries matched correctly
        print(f"  Evaluated {num_queries} queries  |  "
              f"{queries_with_results} had results  |  "
              f"{queries_no_qrels} had no qrels (skipped)")

        # Average across all queries
        summary = {
            metric: round(sum(vals) / len(vals), 4) if vals else 0.0
            for metric, vals in scores.items()
        }
        summary["num_queries"]           = num_queries
        summary["queries_with_results"]  = queries_with_results
        summary["queries_with_no_qrels"] = queries_no_qrels

        return summary


if __name__ == "__main__":
    # Quick sanity check with toy data
    evaluator = Evaluator()

    # Fake ranked results — doc_1 is relevant, doc_2 is not
    fake_results = {
        "q1": [("doc_1", 0.95), ("doc_2", 0.80), ("doc_3", 0.60)],
        "q2": [("doc_4", 0.70), ("doc_1", 0.50)],
    }
    fake_qrels = {
        "q1": {"doc_1": 1},
        "q2": {"doc_4": 1, "doc_5": 1},
    }

    metrics = evaluator.evaluate(fake_results, fake_qrels, k_values=[1, 5, 10])

    print("\nSanity check metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")