import os
from typing import List
from langchain_core.documents import Document

from src.indexing.document_processor import DocumentProcessor
from src.indexing.neo4j_graph_indexer import Neo4jGraphIndexer
from src.retrieval.neo4j_graph_retriever import Neo4jGraphRetriever
from src.config import RAGConfig


class RAGModel:
    """
    Base RAG model using:
    - Neo4j Graph Indexing
    - Neo4jGraphRetriever for contextual search
    """

    def __init__(self, config: RAGConfig, documents: List[Document]):
        self.config = config
        self.chunk_docs = documents or []

        from src.retrieval.faiss_retriever import LocalFAISSRetriever

        # Embedder
        self.doc_processor = DocumentProcessor(self.config)
        self.embedder = self.doc_processor.get_embeddings()

        # Local retriever para NAIVE RAG
        self.retriever = LocalFAISSRetriever(
            self.embedder,
            self.chunk_docs,
            top_k=self.config.num_retrieved_docs
        )

    def retrieve_context(self, query: str) -> List[Document]:
        """Retrieve relevant chunks using Neo4j graph similarity search."""
        return self.retriever.retrieve(query)

    def generate_response(self, query: str) -> str:
        """Must be implemented by the child RAG class (e.g., GPTRAG or LocalRAG)."""
        raise NotImplementedError("generate_response() must be implemented by a subclass.")

    def close(self):
        """Safe close of Neo4j connections."""
        try:
            self.indexer.close()
            self.retriever.close()
        except:
            pass

    def __del__(self):
        """Destructor ensures Neo4j connections are closed."""
        self.close()
