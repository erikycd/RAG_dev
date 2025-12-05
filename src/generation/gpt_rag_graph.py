from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.retrieval.rag_model import RAGModel
from src.retrieval.neo4j_graph_retriever import Neo4jGraphRetriever
from src.indexing.document_processor import DocumentProcessor


class GPTRAG(RAGModel):
    """
    Versión GPT-RAG para Graph-RAG.

    - Usa Neo4jGraphRetriever para:
      * recuperar chunks por similitud
      * extraer metadatos (autor, año, doi, etc.)
    - Aplica reglas estrictas: solo responde con lo que está en el contexto.
    """

    def __init__(self, config, documents):
        # Inicializa como en Naive (RAGModel crea embeddings y FAISS),
        # aunque aquí NO usamos FAISS; no pasa nada si queda sin usar.
        super().__init__(config, documents)

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=config.temperature,
            max_tokens = 1024
        )

        # Memoria simple basada en lista
        self.memory = []

        # Retriever basado en Neo4j (índice vectorial + grafo)
        # Reutilizamos el mismo embedder que RAGModel ya usa.
        self.graph_retriever = Neo4jGraphRetriever(
            config=config,
            embedder=self.embedder
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Eres un asistente experto en Graph RAG.\n\n"
                "Contexto recuperado del documento (fragmentos reales):\n"
                "{context}\n\n"
                "Pregunta del usuario:\n"
                "{question}\n\n"
                "Reglas estrictas:\n"
                "- Solo puedes responder utilizando la información del contexto.\n"
                "- Si la respuesta NO está en el contexto, responde EXACTAMENTE:\n"
                "  \"No encontrado en el documento.\"\n"
                "- Incluye siempre:\n"
                "   * Un texto corto con la respuesta directa\n"
                "   * Entre 1 y 2 citas textuales EXACTAS del contexto\n"
                "   * La página (page_number) de donde proviene la información\n"
                "- No inventes nada que no esté explícito en el texto.\n"
                "- No incluyas opiniones ni resúmenes innecesarios.\n"
                "- Responde en español.\n"
            )
        )

    def generate_response(self, query: str) -> str:
        q = query.lower()

        # ---  Consultas a metadatos (sin embeddings) ---
        meta_fields = {
            "autor": "author_real",
            "autores": "author_real",
            "escribió": "author_real",

            "año": "year",
            "publicó": "year",
            "publicación": "year",
            "fecha": "year",

            "doi": "doi",

            "revista": "journal",
            "volumen": "volume",
            "numero": "issue",
            "número": "issue",

            "palabras clave": "keywords",
        }

        for keyword, field in meta_fields.items():
            if keyword in q:
                try:
                    values = self.graph_retriever.retrieve_metadata(field)
                except Exception:
                    values = None

                if values:
                    return ", ".join(str(v) for v in values)
                return f"No encontrado en metadatos ({field})."

        # --- Recuperación normal basada en embeddings + grafo ---
        try:
            retrieved_docs = self.graph_retriever.retrieve(
                query,
                k=self.config.num_retrieved_docs
            )
        except Exception:
            # Si algo falla en Neo4j, sé estricto:
            return "No encontrado en el documento."

        if not retrieved_docs:
            return "No encontrado en el documento."

        # Agregar contexto con metadata útil
        context = ""
        for doc in retrieved_docs:
            page = doc.metadata.get("page_number", "¿?")
            context += f"[Página {page}]\n{doc.page_content}\n\n"

        # Construcción del prompt final
        formatted_prompt = self.prompt.format(
            context=context,
            question=query
        )

        # Historial (opcional)
        messages = []
        for m in self.memory:
            role = "user" if m["role"] == "user" else "assistant"
            messages.append({
                "role": role,
                "content": m["content"]
            })

        messages.append({"role": "user", "content": formatted_prompt})

        # Llamada al LLM
        response = self.llm.invoke(messages).content

        # Guardar historial
        self.memory.append({"role": "user", "content": query})
        self.memory.append({"role": "assistant", "content": response})

        return response
