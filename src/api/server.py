from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# =========================
# Logging
# =========================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# =========================
# Configuración
# =========================
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1/chat/completions")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "ESIA3")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8001"))

if not PROXY_API_KEY or PROXY_API_KEY == "ESIA3":
    logger.warning("⚠️  PROXY_API_KEY usando valor por defecto. Establece en .env para producción.")

app = FastAPI(
    title="LLM Proxy - RAG Distribuido",
    description="Proxy OpenAI-compatible para LM Studio",
    version="1.0.0"
)

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restricir en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Middleware de Auth
# =========================
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Permitir /health sin autenticación
    if request.url.path == "/health":
        return await call_next(request)
    
    auth = request.headers.get("Authorization")

    if not auth or not auth.startswith("Bearer "):
        logger.warning(f"Intento sin API key desde {request.client.host}")
        raise HTTPException(status_code=401, detail="Missing API key")

    token = auth.split(" ")[1]
    if token != PROXY_API_KEY:
        logger.warning(f"API key inválida desde {request.client.host}")
        raise HTTPException(status_code=403, detail="Invalid API key")

    return await call_next(request)

# =========================
# Health Check
# =========================
@app.get("/health")
async def health_check():
    """Verifica que el proxy y LM Studio están disponibles (sin auth)"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            # LM Studio requiere un request válido, pero sin messages innecesarios
            # Intentar con payload mínimo
            test_payload = {
                "model": "unsloth/deepseek-r1-distill-qwen-7b",
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0.1,
                "max_tokens": 1
            }
            
            try:
                resp = await client.post(
                    LMSTUDIO_URL,
                    json=test_payload,
                    timeout=5
                )
                # Si LM Studio responde (200 o 400 con error del modelo, pero está up)
                lm_ok = resp.status_code in [200, 400, 422]
                lm_status = "online" if lm_ok else "offline"
                
                if resp.status_code != 200:
                    logger.debug(f"LM Studio responded with {resp.status_code}: {resp.text[:100]}")
            except Exception as inner_e:
                logger.debug(f"LM Studio connection error: {inner_e}")
                lm_ok = False
                lm_status = "offline"
        
        return {
            "status": "healthy" if lm_ok else "degraded",
            "proxy": "online",
            "lmstudio": lm_status
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "proxy": "online",
            "lmstudio": "offline",
            "error": str(e)
        }

# =========================
# Endpoint OpenAI-compatible
# =========================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Endpoint compatible con OpenAI API.
    Reenvía solicitudes a LM Studio con reintentos.
    """
    try:
        payload = await request.json()
        
        logger.info(f"Solicitud: modelo={payload.get('model')}, "
                   f"messages={len(payload.get('messages', []))}")

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                LMSTUDIO_URL,
                json=payload,
                timeout=300
            )
        
        logger.info(f"Respuesta: status={response.status_code}")
        
        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )
        
    except httpx.TimeoutException:
        logger.error("Timeout conectando a LM Studio")
        raise HTTPException(status_code=504, detail="LM Studio timeout")
    except httpx.ConnectError:
        logger.error("No se pudo conectar a LM Studio")
        raise HTTPException(status_code=503, detail="LM Studio unavailable")
    except Exception as e:
        logger.error(f"Error en chat_completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Startup
# =========================
@app.on_event("startup")
async def startup():
    logger.info(f"🚀 Proxy iniciado en puerto {PROXY_PORT}")
    logger.info(f"📡 LM Studio: {LMSTUDIO_URL}")
