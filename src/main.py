import sys
from dotenv import load_dotenv

load_dotenv()

from src.config import RAGConfig
from src.indexing.document_processor import DocumentProcessor

# Intentaremos importar el indexador de Neo4j.
# Si no está instalado o hay error de import, marcamos bandera.
try:
    from src.indexing.neo4j_graph_indexer import Neo4jGraphIndexer
    NEO4J_IMPORT_OK = True
    NEO4J_IMPORT_ERROR = None
except Exception as e:
    Neo4jGraphIndexer = None
    NEO4J_IMPORT_OK = False
    NEO4J_IMPORT_ERROR = e


def build_naive_rag(config, documents, model_name: str):
    """Construye el RAG-Naive (FAISS + GPT o LOCAL)."""
    from src.generation.gpt_rag import GPTRAG as NaiveGPTRAG
    from src.generation.local_rag import LocalRAG
    from src.generation.remote_rag import RemoteRAG

    model_name = model_name.upper()

    if model_name == "LOCAL":
        print("Usando modelo LOCAL con RAG-Naive")
        rag = LocalRAG(config, documents)
        used_model = "LOCAL"
    elif model_name == "REMOTE":
        print("Usando modelo REMOTE con RAG-Naive")
        rag = RemoteRAG(config, documents)
        used_model = "REMOTE"
    else:
        print("Usando modelo GPT con RAG-Naive")
        rag = NaiveGPTRAG(config, documents)
        used_model = "GPT"

    return rag, used_model


def build_graph_rag(config, doc_processor, documents):
    """Construye el Graph-RAG (Neo4j). Puede lanzar excepción si falla."""
    if not NEO4J_IMPORT_OK:
        raise RuntimeError(
            f"No se pudo importar Neo4jGraphIndexer: {NEO4J_IMPORT_ERROR}"
        )

    from src.generation.gpt_rag_graph import GPTRAG as GraphGPTRAG

    print("Indexando chunks en Neo4j...")
    indexer = Neo4jGraphIndexer(config, doc_processor.get_embeddings())
    indexer.index_documents(documents)
    print("Indexación en Neo4j completada.")

    rag = GraphGPTRAG(config, documents)
    return rag, indexer


def main(architecture: str = "naive", model: str = "gpt"):
    """
    architecture: 'naive' o 'graph'
    model: 'gpt' o 'local' (para naive). En graph sólo usamos GPT.
    """

    config = RAGConfig()

    print("Cargando y chunking del PDF...")
    doc_processor = DocumentProcessor(config)
    documents = doc_processor.load_documents("./data/raw/Article_1.pdf")

    architecture = architecture.lower()
    model = model.lower()

    rag = None
    indexer = None
    effective_arch = architecture
    effective_model = model

    # 1) Intentar Graph-RAG si se pidió
    if architecture == "graph":
        try:
            print("\n Modo solicitado: GRAPH-RAG")
            rag, indexer = build_graph_rag(config, doc_processor, documents)
            effective_arch = "graph"
            effective_model = "gpt"  # por ahora sólo GPT en graph
        except Exception as e:
            print(f"\n No se pudo iniciar Graph-RAG por el error:")
            print(f"   {e}")
            print(" Haciendo fallback automático a RAG-Naive (FAISS).\n")
            architecture = "naive"

    # 2) Si no se pudo Graph o se pidió Naive directamente
    if architecture == "naive":
        print("\n Modo: RAG-Naive")
        rag, effective_model = build_naive_rag(config, documents, model)
        effective_arch = "naive"

    print("\n" + "=" * 60)
    print(f"Arquitectura en uso : {effective_arch.upper()}")
    print(f"Modelo de lenguaje  : {effective_model.upper()}")
    print("Escribe tu pregunta o 'quit' para salir.")
    print("=" * 60 + "\n")

    # Bucle de conversación
    try:
        while True:
            query = input("\nUsuario: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "salir"]:
                print("\n ¡Hasta pronto!")
                break

            print("Pensando...")
            try:
                response = rag.generate_response(query)
            except Exception as e:
                print(f"\n Error al procesar la consulta: {e}")
                continue

            print(f"\nAsistente: {response}")

    finally:
        # Limpieza del grafo sólo si se usó Graph-RAG
        if indexer is not None:
            try:
                print("\n Limpiando grafo Neo4j...")
                indexer.clear_graph()
            except Exception as e:
                print(f" Error durante la limpieza de Neo4j: {e}")


if __name__ == "__main__":
    """
    Uso:

    python main.py              → naive + GPT (por defecto)
    python main.py naive        → naive + GPT
    python main.py naive local  → naive + LOCAL
    python main.py naive remote → naive + REMOTE
    python main.py graph        → graph + GPT (con fallback a naive)
    """
    args = sys.argv[1:]

    if len(args) == 0:
        arch = "naive"
        model = "gpt"
    elif len(args) == 1:
        arch = args[0]
        model = "gpt"
    else:
        arch = args[0]
        model = args[1]

    main(arch, model)

