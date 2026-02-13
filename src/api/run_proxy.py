import uvicorn
from src.api.server import app
from src.config import RAGConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    config = RAGConfig()
    
    logger.info("=" * 60)
    logger.info("Iniciando Proxy RAG Distribuido")
    logger.info("=" * 60)
    logger.info(f"Puerto        : {config.proxy_port}")
    logger.info(f"LM Studio URL : {config.lmstudio_url}")
    logger.info(f"API Key       : {config.proxy_api_key[:10]}***")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.proxy_port,
        log_level="info"
    )
