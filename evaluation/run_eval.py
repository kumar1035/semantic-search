# evaluation/run_eval.py

import argparse
import json
import os
import time
from evaluation.dataset_loader import DatasetLoader
from evaluation.indexer_bridge  import IndexerBridge
from evaluation.query_runner    import QueryRunner
from evaluation.evaluator       import Evaluator


MODES           = ["dense", "sparse", "hybrid", "full"]
DISPLAY_METRICS = ["NDCG@10", "MAP@100", "Recall@100", "P@10", "MRR"]

# All supported datasets — add more here later if needed
AVAILABLE_DATASETS = {
    "scifact":  "data/scifact",
    "nfcorpus": "data/nfcorpus",
}


def print_table(results: dict, title: str = ""):
    col_w  = 14
    header = f"{'Mode':<10}" + "".join(f"{m:>{col_w}}" for m in DISPLAY_METRICS)
    if title:
        print(f"\n  {title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for mode, metrics in results.items():
        row = f"{mode:<10}"
        for m in DISPLAY_METRICS:
            val = metrics.get(m, 0.0)
            row += f"{val:>{col_w}.4f}"
        print(row)
    print("=" * len(header))


def print_comparison_table(all_dataset_results: dict):
    """
    Print a single comparison table across all datasets.
    Shows NDCG@10 and MRR side by side for each dataset.
    """
    datasets = list(all_dataset_results.keys())
    modes    = list(list(all_dataset_results.values())[0].keys())

    print("\n" + "=" * 80)
    print("CROSS-DATASET COMPARISON — full pipeline mode")
    print("=" * 80)

    # Header
    header = f"{'Dataset':<14}" + "".join(
        f"{'NDCG@10':>12}{'MRR':>10}{'MAP@100':>10}"
    )
    print(f"{'Dataset':<14}{'NDCG@10':>12}{'MRR':>10}{'MAP@100':>10}")
    print("-" * 46)

    for dataset, mode_results in all_dataset_results.items():
        # use "full" mode results for comparison, fallback to first mode
        metrics = mode_results.get("full", list(mode_results.values())[0])
        ndcg    = metrics.get("NDCG@10", 0.0)
        mrr     = metrics.get("MRR", 0.0)
        map_    = metrics.get("MAP@100", 0.0)
        print(f"{dataset:<14}{ndcg:>12.4f}{mrr:>10.4f}{map_:>10.4f}")

    print("=" * 46)


def run_single_dataset(dataset_name: str, dataset_path: str, args) -> dict:
    """Run full eval pipeline for one dataset. Returns mode→metrics dict."""

    print(f"\n{'#'*60}")
    print(f"  DATASET: {dataset_name.upper()}")
    print(f"{'#'*60}")

    # 1 — load
    print("\n[1/4] Loading dataset...")
    loader  = DatasetLoader(dataset_path)
    corpus  = loader.load_corpus()
    queries = loader.load_queries()
    qrels   = loader.load_qrels()

    # 2 — index
    if not args.skip_index:
        print("\n[2/4] Indexing corpus...")
        bridge = IndexerBridge(args.config)
        # pass dataset_name so fake paths are e.g. nfcorpus://doc_id
        bridge.index_corpus(corpus, batch_size=64, dataset_name=dataset_name)
    else:
        print("\n[2/4] Skipping indexing (--skip-index)")

    # 3 — run queries
    print("\n[3/4] Running queries...")
    runner    = QueryRunner(args.config)
    evaluator = Evaluator()

    modes_to_run     = MODES if args.mode == "all" else [args.mode]
    all_mode_results = {}

    for mode in modes_to_run:
        print(f"\n  Mode: {mode}")
        t0             = time.time()
        ranked_results = runner.run(queries, top_k=args.top_k, mode=mode)
        elapsed        = time.time() - t0

        metrics                  = evaluator.evaluate(ranked_results, qrels, k_values=[1, 5, 10, 100])
        metrics["query_time_s"]  = round(elapsed, 2)
        all_mode_results[mode]   = metrics

        print(f"  NDCG@10={metrics.get('NDCG@10', 0):.4f}  "
              f"MAP@100={metrics.get('MAP@100', 0):.4f}  "
              f"MRR={metrics.get('MRR', 0):.4f}")

    # 4 — per-dataset table
    print(f"\n[4/4] Results for {dataset_name.upper()}")
    print_table(all_mode_results, title=f"EVALUATION RESULTS — {dataset_name} (pytrec_eval)")

    return all_mode_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic search on BEIR datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "nfcorpus"],
        choices=list(AVAILABLE_DATASETS.keys()),
        help="Which datasets to evaluate. e.g. --datasets scifact nfcorpus"
    )
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--top-k",      default=100, type=int)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--mode",       default="all",
                        help="dense | sparse | hybrid | full | all")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    all_dataset_results = {}

    for dataset_name in args.datasets:
        dataset_path = AVAILABLE_DATASETS[dataset_name]

        if not os.path.exists(dataset_path):
            print(f"\n[WARNING] Dataset folder not found: {dataset_path} — skipping {dataset_name}")
            continue

        results = run_single_dataset(dataset_name, dataset_path, args)
        all_dataset_results[dataset_name] = results

        # save per-dataset report
        report_path = f"results/eval_{dataset_name}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {report_path}")

    # cross-dataset comparison (only if more than one dataset ran)
    if len(all_dataset_results) > 1:
        print_comparison_table(all_dataset_results)

    # save combined report
    combined_path = "results/eval_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_dataset_results, f, indent=2)
    print(f"\nCombined report saved → {combined_path}")


if __name__ == "__main__":
    main()