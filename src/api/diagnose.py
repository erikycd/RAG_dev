import httpx
import json
import sys

def test_lmstudio():
    """Diagnostica conexión a LM Studio"""
    print("\n" + "=" * 60)
    print("DIAGNOSTICO: LM Studio")
    print("=" * 60)
    
    url = "http://127.0.0.1:1234/v1/chat/completions"
    
    # Test 1: Conectividad básica
    print("\n[1] Verificando conectividad básica...")
    try:
        response = httpx.get("http://127.0.0.1:1234", timeout=5)
        print("✅ Puedo alcanzar http://127.0.0.1:1234")
    except Exception as e:
        print(f"❌ No puedo alcanzar LM Studio: {e}")
        print("   → Asegúrate que LM Studio está corriendo")
        print("   → Verifica que el puerto es 1234")
        return False
    
    # Test 2: Endpoint /v1/chat/completions con diferentes payloads
    test_cases = [
        ("model=default", {
            "model": "default",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }),
        ("model=unsloth/deepseek-r1-distill-qwen-7b", {
            "model": "unsloth/deepseek-r1-distill-qwen-7b",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }),
        ("sin max_tokens", {
            "model": "unsloth/deepseek-r1-distill-qwen-7b",
            "messages": [{"role": "user", "content": "test"}]
        }),
    ]
    
    for desc, payload in test_cases:
        print(f"\n[2.{test_cases.index((desc, payload)) + 1}] Testing con {desc}...")
        print(f"   Payload: {json.dumps(payload, indent=2)[:100]}...")
        
        try:
            response = httpx.post(url, json=payload, timeout=5)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print("   ✅ Request exitoso")
                data = response.json()
                if "choices" in data:
                    print(f"   Respuesta: {data['choices'][0]['message']['content'][:50]}...")
                return True
            else:
                print(f"   ⚠️ Status {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except httpx.TimeoutException:
            print(f"   ❌ Timeout - LM Studio está lento o no responde")
        except httpx.ConnectError as e:
            print(f"   ❌ Connection error: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return False

def test_proxy():
    """Diagnostica conexión al Proxy"""
    print("\n" + "=" * 60)
    print("DIAGNOSTICO: Proxy FastAPI")
    print("=" * 60)
    
    base_url = "http://localhost:8001"
    
    print("\n[1] Verificando conectividad al proxy...")
    try:
        response = httpx.get(f"{base_url}/health", timeout=5)
        print(f"✅ Proxy responde con status {response.status_code}")
        data = response.json()
        print(f"   Status: {data.get('status')}")
        print(f"   Proxy: {data.get('proxy')}")
        print(f"   LM Studio: {data.get('lmstudio')}")
        return True
    except Exception as e:
        print(f"❌ No puedo conectar al proxy: {e}")
        print("   → Ejecuta: python run_proxy.py")
        return False

def main():
    print("\n" + "=" * 70)
    print("🔍 DIAGNÓSTICO COMPLETO - RAG PROXY")
    print("=" * 70)
    
    # Test LM Studio
    lm_ok = test_lmstudio()
    
    # Test Proxy
    proxy_ok = test_proxy()
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"LM Studio: {'✅ OK' if lm_ok else '❌ FALLÓ'}")
    print(f"Proxy:     {'✅ OK' if proxy_ok else '❌ FALLÓ'}")
    print("=" * 70 + "\n")
    
    if not lm_ok:
        print("PRÓXIMOS PASOS:")
        print("1. Abre LM Studio GUI")
        print("2. Carga un modelo (e.g., Deepseek R1 Distill)")
        print("3. Ve a 'Server' → Start Server")
        print("4. Verifica que el endpoint muestre: http://127.0.0.1:1234/v1")
        print("5. Ejecuta este script de nuevo\n")
    
    if not proxy_ok:
        print("PRÓXIMOS PASOS:")
        print("1. Abre una terminal")
        print("2. Ejecuta: python run_proxy.py")
        print("3. Espera el mensaje: 'Proxy iniciado en puerto 8001'")
        print("4. Ejecuta este script de nuevo\n")

if __name__ == "__main__":
    main()
