from dataclasses import dataclass
from typing import Literal
import os
from dotenv import load_dotenv

# Cargar archivo .env automáticamente
load_dotenv()

@dataclass
class RAGConfig:
    # Credenciales
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "neo4jpassword")

    openai_api_key: str = os.getenv("OPENAI_API_KEY")

    # Modelos
    embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    device: str = "cpu"

    # Text Splitter
    chunk_size: int = 512
    chunk_overlap: int = 50

    # RAG
    num_retrieved_docs: int = 12
    temperature: float = 0.1
    model_mode: Literal["GPT", "LOCAL"] = "GPT"

    # Grafo
    edge_similarity_threshold: float = 0.75
    edge_top_k: int = 5

    # LLM REMOTO (Cliente) - Apunta al PROXY, no directamente a LM Studio
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "ESIA3")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "unsloth/deepseek-r1-distill-qwen-7b")

    # === PROXY SERVER ===
    proxy_port: int = int(os.getenv("PROXY_PORT", "8001"))
    proxy_api_key: str = os.getenv("PROXY_API_KEY", "ESIA3")
    lmstudio_url: str = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")

