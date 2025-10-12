import os
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from src.retrieval.rag_model import RAGModel
from src.config import RAGConfig

class GPTRAG(RAGModel):
    """RAG implementation using GPT-4."""
    def __init__(self, config: RAGConfig, documents):
        super().__init__(config, documents)
        self.llm = self._init_gpt()
    
    def _init_gpt(self):
        """Initialize GPT model."""
        return init_chat_model(
            model = "gpt-4.1-nano",
            model_provider = "openai",
            openai_api_key = os.environ["OPENAI_API_KEY"],
            temperature = self.config.temperature
        )
    
    def generate_response(self, query: str) -> str:
        context_docs = self.retrieve_context(query)
        context = "\n\n".join(doc.page_content for doc in context_docs)
        template = """You're a helpful assistant. Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question}"""
        prompt = PromptTemplate(
            template = template,
            input_variables = ["context", "question"]
        )
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": query})
        return response.content
