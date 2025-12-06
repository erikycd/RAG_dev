import os
import numpy as np
from neo4j import GraphDatabase, Driver
from typing import List, Dict
from langchain_core.documents import Document

from src.config import RAGConfig

# Nota: Neo4j Python Driver debe estar instalado (pip install neo4j)

class Neo4jGraphRetriever:
    """
    Retriever que usa el índice vectorial de Neo4j + expansión por grafo
    (k-hop) a través de las relaciones :SIMILAR_TO.
    """

    def __init__(self, config: RAGConfig, embedder):
        self.config = config
        self.embedder = embedder

        # Conexión a Neo4j (usa las variables de entorno)
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "neo4jpassword")
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def retrieve(self, query: str, k: int = None, hops: int = 1) -> List[Document]:
        k = k or self.config.num_retrieved_docs

        #Embedding de la query
        qvec = self.embedder.embed_query(query)

        # Buscar muchos nodos en el grafo (Top-20 inicial)
        initial_k = max(k * 5, 20)

        cypher_query = f"""
        CALL db.index.vector.queryNodes(
            'chunk_embeddings',
            {initial_k},
            $embedding
        ) YIELD node AS retrievedNode, score AS vectorScore
        RETURN retrievedNode AS node, vectorScore
        ORDER BY vectorScore DESC
        LIMIT {initial_k}
        """

        with self.driver.session() as session:
            records = session.run(cypher_query, embedding=qvec).data()

        # Reranking semántico local (cosine similarity)
        reranked = []
        for rec in records:
            node = rec["node"]
            metadata = dict(node)
            text = metadata.pop("text", "")
            emb = np.array(metadata.pop("embedding", []), dtype="float32")

            # sim(query, embedding)
            score = float(np.dot(qvec, emb) / (np.linalg.norm(qvec) * np.linalg.norm(emb)))

            reranked.append((score, text, metadata))

        # Ordenar y filtrar los mejores K
        reranked.sort(key=lambda x: x[0], reverse=True)
        reranked = reranked[:k]

        # Convertir a documentos LangChain
        results = []
        for score, text, metadata in reranked:
            metadata["rerank_score"] = score
            results.append(Document(page_content=text, metadata=metadata))

        return results

    def retrieve_metadata(self, field: str):
        query = f"""
        MATCH (c:Chunk)
        WHERE c.{field} IS NOT NULL
        RETURN DISTINCT c.{field} AS value
        LIMIT 5
        """
        with self.driver.session() as session:
            return [rec["value"] for rec in session.run(query).data()]

    def close(self):
        self.driver.close()
