from openai import OpenAI
from src.retrieval.rag_model import RAGModel
from src.config import RAGConfig

class LocalRAG(RAGModel):
    """RAG implementation using a local LLM."""
    def __init__(self, config: RAGConfig, documents):
        super().__init__(config, documents)
        self.llm = self._init_local_llm()
    
    def _init_local_llm(self):
        """Initialize local LLM."""
        return OpenAI(
            base_url = "http://localhost:1234/v1",
            api_key = "lm-studio"
        )
    
    def generate_response(self, query: str) -> str:
        context_docs = self.retrieve_context(query)
        context = "\n\n".join(doc.page_content for doc in context_docs)
        template = """You're a helpful assistant. Answer the question based on the context below.\n\n{context}\n\nQuestion: {question}\n\nAnswer:"""
        response = self.llm.chat.completions.create(
            model = 'unsloth/deepseek-r1-distill-qwen-7b',
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": template.format(context=context, question=query)}
            ],
            temperature = self.config.temperature
        )
        return response.choices[0].message.content
