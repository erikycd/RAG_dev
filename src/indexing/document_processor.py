from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import RAGConfig
from typing import List

class DocumentProcessor:
    """Handle document loading and processing."""
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,
            chunk_overlap = config.chunk_overlap
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        """Load and split documents from a PDF file."""
        loader = PyPDFLoader(file_path)
        return loader.load_and_split(text_splitter = self.text_splitter)
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize and return the embedding model."""
        return HuggingFaceEmbeddings(
            model_name = self.config.embedding_model,
            model_kwargs = {'device': self.config.device},
            encode_kwargs = {'normalize_embeddings': False}
        )
