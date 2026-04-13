# app.py

import json
import os
import time
from flask import Flask, render_template, request

from searcher.search_engine import SearchEngine
from evaluation.dataset_loader import DatasetLoader

app = Flask(__name__)

# ── load search engine once ──────────────────────────────────────────────────
print("Loading search engine...")
engine = SearchEngine("config.yaml")
print("Search engine ready.")


# ── load dataset queries once ────────────────────────────────────────────────
def load_dataset_queries():
    result   = {}
    datasets = {
        "scifact":  "data/scifact",
        "nfcorpus": "data/nfcorpus",
    }
    for name, path in datasets.items():
        if os.path.exists(path):
            try:
                loader       = DatasetLoader(path)
                result[name] = loader.load_queries()
                print(f"Loaded {len(result[name])} queries from {name}")
            except Exception as e:
                print(f"Could not load {name}: {e}")
                result[name] = {}
        else:
            result[name] = {}
    return result


DATASET_QUERIES = load_dataset_queries()


def load_eval_results():
    path = "results/eval_all.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_doc_id(filepath):
    if "://" in filepath:
        return filepath.split("://", 1)[1]
    return filepath


def get_dataset_name(filepath):
    if "scifact://"  in filepath: return "scifact"
    if "nfcorpus://" in filepath: return "nfcorpus"
    return "filesystem"


def get_icon(filepath):
    if "scifact://"  in filepath: return "🔬"
    if "nfcorpus://" in filepath: return "🏥"
    ext = filepath.rsplit(".", 1)[-1].lower() if "." in filepath else ""
    return {"pdf": "📄", "docx": "📝", "txt": "📃",
            "pptx": "📊", "xlsx": "📋", "py": "🐍"}.get(ext, "📄")


def get_snippet(r):
    for key in ["preview", "chunk_text", "text"]:
        val = r.get(key, "")
        if val and str(val).strip():
            s = str(val).strip()
            return s[:220].rsplit(" ", 1)[0] + "..." if len(s) > 220 else s
    return "No preview available."


def get_score(r):
    for key in ["rerank_score", "rrf_score", "dense_score"]:
        val = r.get(key)
        if val is not None:
            return round(float(val), 4)
    return 0.0


def find_matched_queries(user_query):
    """
    Find dataset queries that share words with the user query.
    Returns up to 6 per dataset.
    """
    user_words = set(
        w.lower().strip(".,?!") for w in user_query.split()
        if len(w) > 3
    )
    if not user_words:
        return [], []

    scifact_matches  = []
    nfcorpus_matches = []

    for dataset_name, queries in DATASET_QUERIES.items():
        for qid, qtext in queries.items():
            q_words = set(
                w.lower().strip(".,?!") for w in qtext.split()
                if len(w) > 3
            )
            overlap = len(user_words & q_words)
            if overlap > 0:
                entry = {
                    "query_id":   qid,
                    "query_text": qtext,
                    "overlap":    overlap,
                }
                if dataset_name == "scifact":
                    scifact_matches.append(entry)
                else:
                    nfcorpus_matches.append(entry)

    scifact_matches.sort( key=lambda x: x["overlap"], reverse=True)
    nfcorpus_matches.sort(key=lambda x: x["overlap"], reverse=True)

    return scifact_matches[:6], nfcorpus_matches[:6]


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html",
        scifact_count  = len(DATASET_QUERIES.get("scifact",  {})),
        nfcorpus_count = len(DATASET_QUERIES.get("nfcorpus", {})),
    )


@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query", "").strip()
    top_k = int(request.form.get("top_k", 10))

    if not query:
        return render_template("index.html",
            error          = "Please enter a search query.",
            scifact_count  = len(DATASET_QUERIES.get("scifact",  {})),
            nfcorpus_count = len(DATASET_QUERIES.get("nfcorpus", {})),
        )

    try:
        t0      = time.time()
        output  = engine.search(query, top_k=top_k)
        elapsed = round(time.time() - t0, 3)

        raw_results = output.get("results",    [])
        query_info  = output.get("query_info", {})

        results = []
        for r in raw_results:
            filepath = r.get("filepath", "")
            results.append({
                "doc_id":   extract_doc_id(filepath),
                "filepath": filepath,
                "score":    get_score(r),
                "snippet":  get_snippet(r),
                "icon":     get_icon(filepath),
                "dataset":  get_dataset_name(filepath),
            })

        scifact_matches, nfcorpus_matches = find_matched_queries(query)

        return render_template("results.html",
            query            = query,
            expanded_query   = query_info.get("expanded",  query),
            results          = results,
            total            = len(results),
            elapsed          = elapsed,
            top_k            = top_k,
            scifact_matches  = scifact_matches,
            nfcorpus_matches = nfcorpus_matches,
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return f"<pre style='color:red;padding:2rem'>{tb}</pre>", 500


@app.route("/dashboard")
def dashboard():
    eval_data = load_eval_results()
    datasets  = []

    for name, mode_results in eval_data.items():
        full = mode_results.get("full", {})
        datasets.append({
            "name":      name,
            "ndcg":      full.get("NDCG@10",    0.0),
            "mrr":       full.get("MRR",         0.0),
            "map":       full.get("MAP@100",     0.0),
            "recall":    full.get("Recall@100",  0.0),
            "precision": full.get("P@10",        0.0),
            "queries":   full.get("num_queries", 0),
            "modes":     mode_results,
        })

    return render_template("dashboard.html", datasets=datasets)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)