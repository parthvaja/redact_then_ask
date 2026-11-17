"""Vector store indexing for RAG (Phase 10).

Wraps ChromaDB for document chunk storage and cosine-similarity retrieval.
"""

from __future__ import annotations

import logging

from rta.config import get_config

logger = logging.getLogger(__name__)


class VectorStore:
    """Chunk embeddings stored in a ChromaDB collection."""

    def __init__(self, doc_hash: str, chunks: list[dict]) -> None:
        self.doc_hash = doc_hash
        self.chunks = chunks
        self._collection = None
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            cfg = get_config()
            model_name = cfg.rag.embed_model
            self._embedder = SentenceTransformer(model_name)
            logger.debug("Loaded embedding model %s", model_name)
        return self._embedder

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        import chromadb

        cfg = get_config()
        persist_dir = str(cfg.paths.cache) + "/chroma"
        client = chromadb.PersistentClient(path=persist_dir)
        name = f"doc_{self.doc_hash[:16]}"
        self._collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        if self._collection.count() == 0:
            self._index_chunks()
        return self._collection

    def _index_chunks(self) -> None:
        embedder = self._get_embedder()
        texts = [c["text"] for c in self.chunks]
        embeddings = embedder.encode(texts).tolist()
        ids = [str(c["chunk_id"]) for c in self.chunks]
        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
        )
        logger.info("Indexed %d chunks into ChromaDB", len(texts))

    def query(self, question: str, top_k: int = 4) -> list[dict]:
        """Retrieve *top_k* chunks most similar to *question*."""
        embedder = self._get_embedder()
        q_embedding = embedder.encode([question]).tolist()
        collection = self._get_collection()

        n = min(top_k, len(self.chunks))
        if n == 0:
            return []

        results = collection.query(
            query_embeddings=q_embedding,
            n_results=n,
        )

        retrieved: list[dict] = []
        for i, doc_id in enumerate(results["ids"][0]):
            retrieved.append(
                {
                    "chunk_id": int(doc_id),
                    "text": results["documents"][0][i],
                    "distance": (results["distances"][0][i] if results.get("distances") else None),
                }
            )

        return retrieved
