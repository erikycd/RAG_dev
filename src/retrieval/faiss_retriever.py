import numpy as np
from typing import List
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity


class LocalFAISSRetriever:
    """
    Lightweight FAISS-like retriever using numpy + cosine similarity.
    Designed for Naive RAG without using Neo4j.
    """

    def __init__(self, embedder, documents: List[Document], top_k: int = 12):
        self.embedder = embedder
        self.documents = documents
        self.top_k = top_k

        # Pre-calculate embeddings for faster retrieval
        texts = [d.page_content for d in documents]
        self.embeddings = embedder.embed_documents(texts)
        self.embeddings = np.array(self.embeddings, dtype=np.float32)

    def retrieve(self, query: str) -> List[Document]:
        """Return top-k most similar chunks."""
        qvec = self.embedder.embed_query(query)
        qvec = np.array(qvec, dtype=np.float32).reshape(1, -1)

        sims = cosine_similarity(qvec, self.embeddings).flatten()
        top_indices = sims.argsort()[::-1][:self.top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc.metadata["similarity_score"] = float(sims[idx])
            results.append(doc)

        return results
