class RAGConfig:
    """Configuration class for RAG system."""
    def __init__(self):
        # Document processing
        self.chunk_size = 3000
        self.chunk_overlap = 200
        # Model parameters
        self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.device = 'cpu'
        self.temperature = 0.7
        # RAG parameters
        self.num_retrieved_docs = 1
