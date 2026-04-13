# indexer/embedder.py

import yaml
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Loads a sentence-transformer model and converts text chunks
    into dense vector embeddings.

    Model upgrade: all-MiniLM-L6-v2  →  BAAI/bge-small-en-v1.5

    Why BGE over MiniLM:
        - MiniLM   : general purpose, fast, 384-dim, NDCG ~0.65 on SciFact
        - BGE-small: retrieval-specific training, 384-dim, NDCG ~0.72 on SciFact
        - Same dimension (384), same API — only the model name changes
        - BGE uses a special instruction prefix for queries (not for documents)
          "Represent this sentence for searching relevant passages: {query}"
          This is handled automatically in embed_single()
    """

    # BGE query instruction prefix — improves retrieval accuracy
    # Applied to queries only, NOT to document chunks during indexing
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, config_path="config.yaml"):
        """
        Load the config and initialize the embedding model.

        Args:
            config_path (str) — path to config.yaml
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_name     = config["embedding_model"]
        self.model_name = model_name

        # detect if we are using a BGE model
        # BGE models need a special prefix on queries (not on documents)
        self.is_bge = "bge" in model_name.lower()

        print(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded — BGE mode: {self.is_bge}")

    def embed_chunks(self, chunks):
        """
        Convert a list of text chunks into dense vector embeddings.
        Used during INDEXING — no query prefix applied here.

        Args:
            chunks (list[str]) — list of text strings to embed

        Returns:
            numpy.ndarray — shape (num_chunks, embedding_dim)
                            384 dimensions for both MiniLM and BGE-small
        """
        embeddings = self.model.encode(
            chunks,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=self.is_bge,  # BGE needs L2 normalization
        )
        return embeddings

    def embed_single(self, text):
        """
        Embed a single query string.
        Used during SEARCH — BGE prefix is applied here if using BGE model.

        Why prefix only on queries:
            BGE was trained with this asymmetric setup.
            Documents are indexed as-is.
            Queries get the instruction prefix so the model knows
            it is searching for relevant passages, not matching exact text.

        Args:
            text (str) — a single query string

        Returns:
            numpy.ndarray — one embedding vector (384 dimensions)
        """
        if self.is_bge:
            text = self.BGE_QUERY_PREFIX + text

        return self.model.encode(
            text,
            normalize_embeddings=True,  # always normalize for BGE
        )


if __name__ == "__main__":
    embedder = Embedder()

    test_chunks = [
        "The quarterly budget report shows increased spending",
        "Machine learning models can understand text semantics",
        "The cat sat on the mat and looked out the window"
    ]

    print("Embedding 3 test chunks...")
    vectors = embedder.embed_chunks(test_chunks)
    print(f"Got {len(vectors)} vectors")
    print(f"Each vector has {len(vectors[0])} dimensions")
    print(f"First vector (first 5 values): {vectors[0][:5]}")

    print("\n--- Single query embedding ---")
    query_vec = embedder.embed_single("budget spending report")
    print(f"Query vector: {len(query_vec)} dimensions")