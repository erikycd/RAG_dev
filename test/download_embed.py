from sentence_transformers import SentenceTransformer
import os

def download_and_save_model(model_name: str, save_path: str):
    print(f"Descargando modelo {model_name}...")
    model = SentenceTransformer(model_name)
    os.makedirs(save_path, exist_ok=True)
    print(f"Guardando modelo en {save_path}...")
    model.save(save_path)
    print("¡Modelo guardado correctamente!")

if __name__ == "__main__":
    model_name = "sentence-transformers/all-mpnet-base-v2"
    save_path = "./data/embeddings/all-mpnet-base-v2"
    download_and_save_model(model_name, save_path)
