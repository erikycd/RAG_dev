from langchain.vectorstores import FAISS
from src.indexing.document_processor import DocumentProcessor
from langchain_core.documents import Document
from src.config import RAGConfig
from typing import List

class RAGModel:
    """Base RAG model class."""
    def __init__(self, config: RAGConfig, documents: List[Document]):
        self.config = config
        self.documents = documents
        self.embeddings = DocumentProcessor(config).get_embeddings()
        self.vector_db = self._create_vector_store()
    
    def _create_vector_store(self) -> FAISS:
        """Create and return a FAISS vector store."""
        return FAISS.from_documents(self.documents, self.embeddings)
    
    def retrieve_context(self, query: str) -> List[Document]:
        """Retrieve relevant context for a query."""
        return self.vector_db.similarity_search(
            query, 
            k = self.config.num_retrieved_docs
        )
    
    def generate_response(self, query: str) -> str:
        """Generate a response (to be implemented by subclasses)."""
        raise NotImplementedError
