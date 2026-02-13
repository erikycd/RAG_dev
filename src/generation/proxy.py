from openai import OpenAI
from src.config import RAGConfig
import logging

logger = logging.getLogger(__name__)

class RemoteGPT:
    """
    Cliente LLM remoto vía backend proxy
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.client = OpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key
        )
        self.model = config.llm_model_name

    def chat(self, messages, temperature: float = None) -> str:
        """
        Args:
            messages: lista de dicts con role/content
            temperature: parámetro de temperatura (usa config si es None)
        """
        temp = temperature if temperature is not None else self.config.temperature
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error en RemoteGPT.chat(): {e}")
            raise
