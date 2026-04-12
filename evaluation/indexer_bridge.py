# evaluation/indexer_bridge.py

import numpy as np
from indexer.chunker import Chunker
from indexer.embedder import Embedder
from indexer.store import Store


class IndexerBridge:
    """
    Feeds the BEIR corpus directly into your existing indexing pipeline.

    The corpus documents are NOT real files on disk — they come from JSONL.
    So we bypass the Crawler/Extractor and inject text directly into
    Chunker → Embedder → Store.

    Each document gets a fake filepath: "{dataset_name}://{doc_id}"
    This lets the Store treat them like any other indexed file,
    and the Evaluator can later match doc_id back from results.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.chunker  = Chunker(chunk_size=500, overlap=50)
        self.embedder = Embedder(config_path)
        self.store    = Store(config_path)

    def index_corpus(self, corpus: dict, batch_size: int = 64, dataset_name: str = "dataset"):
        """
        Index the entire corpus into FAISS + SQLite.

        Args:
            corpus       — {doc_id: {"title": str, "text": str}}
            batch_size   — number of chunks to embed at once (memory control)
            dataset_name — used as prefix for fake file paths e.g. "scifact", "nfcorpus"
        """
        doc_ids = list(corpus.keys())
        total   = len(doc_ids)
        print(f"Indexing {total} documents from [{dataset_name}]...")

        # Clear previous entries for THIS dataset only
        existing_hashes  = self.store.load_hashes()
        prefix           = f"{dataset_name}://"
        existing_entries = [fp for fp in existing_hashes if fp.startswith(prefix)]
        for fp in existing_entries:
            self.store.remove_file_chunks(fp)
        if existing_entries:
            print(f"Cleared {len(existing_entries)} previously indexed [{dataset_name}] documents")

        chunk_buffer = []
        text_buffer  = []

        def flush(chunk_buffer, text_buffer):
            if not chunk_buffer:
                return
            embeddings = self.embedder.embed_chunks(text_buffer)
            embeddings = np.array(embeddings, dtype="float32")
            self.store.add_chunks(chunk_buffer, embeddings)

        for i, doc_id in enumerate(doc_ids, 1):
            doc       = corpus[doc_id]
            full_text = f"{doc['title']} {doc['text']}".strip()
            if not full_text:
                continue

            fake_path = f"{prefix}{doc_id}"
            chunks    = self.chunker.chunk_file(full_text, fake_path)

            for chunk in chunks:
                chunk_buffer.append(chunk)
                text_buffer.append(chunk["text"])

            self.store.save_file_info(fake_path, doc_id, len(chunks))

            if len(chunk_buffer) >= batch_size:
                flush(chunk_buffer, text_buffer)
                chunk_buffer.clear()
                text_buffer.clear()

            if i % 500 == 0:
                print(f"  Indexed {i}/{total}...")

        # flush any remaining chunks
        flush(chunk_buffer, text_buffer)
        print(f"Done. Total vectors: {self.store.get_total_vectors()}")


if __name__ == "__main__":
    from evaluation.dataset_loader import DatasetLoader

    loader = DatasetLoader("data/scifact")
    corpus = loader.load_corpus()

    bridge = IndexerBridge()
    bridge.index_corpus(corpus, batch_size=64, dataset_name="scifact")