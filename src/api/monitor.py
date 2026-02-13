import asyncio
import httpx
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def monitor():
    """Monitorea salud del sistema cada 30 segundos"""
    while True:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Proxy health
                resp1 = await client.get("http://localhost:8001/health")
                proxy_ok = resp1.status_code == 200
                
                # LM Studio health
                resp2 = await client.get("http://127.0.0.1:1234/v1/health")
                lm_ok = resp2.status_code == 200
                
                status = "✅" if (proxy_ok and lm_ok) else "❌"
                logger.info(f"{status} [{datetime.now()}] Proxy:{proxy_ok} LM:{lm_ok}")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        
        await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(monitor())
