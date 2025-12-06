from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from src.retrieval.rag_model import RAGModel


class GPTRAG(RAGModel):
    """
    RAG-Naive con FAISS.
    - Recupera contexto con embeddings desde FAISS
    - Responde usando GPT (o modelo local desde main)
    - Mantiene memoria simple para contexto conversacional
    """

    def __init__(self, config, documents):
        super().__init__(config, documents)
        self.documents = documents
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=config.temperature,
            max_tokens = 1024
        )

        # Memoria simple (igual que Graph)
        self.memory = []

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Eres un asistente basado en Retrieval Augmented Generation.\n\n"
                "Contexto disponible:\n"
                "{context}\n\n"
                "Pregunta:\n"
                "{question}\n\n"
                "Instrucciones:\n"
                "- Responde únicamente con información del texto.\n"
                "- Si no está en el texto, responde:\n"
                "  \"No encontrado en el documento.\"\n"
                "- Incluye referencias a la página cuando sea posible.\n"
                "- Responde en español.\n"
            )
        )

    def generate_response(self, query: str) -> str:
        q = query.lower()

        # Intento 1️⃣: Recuperación basada en embeddings
        retrieved_docs = self.retrieve_context(query)
        retrieved_docs = self.retrieve_context(query)

        # Priorizar secciones con contenido relevante
        retrieved_docs_sorted = sorted(
            retrieved_docs,
            key=lambda d: (
                d.metadata.get("section") not in ["Resumen", "Resultados", "Conclusiones"],
                -len(d.page_content)  # más texto primero
            )
        )

        retrieved_docs = retrieved_docs_sorted[:self.config.num_retrieved_docs]


        if retrieved_docs:
            context = ""
            for doc in retrieved_docs:
                page = doc.metadata.get("page_number", "¿?")
                context += f"[Página {page}] {doc.page_content}\n\n"

            formatted_prompt = self.prompt.format(
                context=context,
                question=query
            )

            messages = []
            for m in self.memory:
                messages.append({
                    "role": m["role"],
                    "content": m["content"]
                })

            messages.append({"role": "user", "content": formatted_prompt})

            response = self.llm.invoke(messages).content

            # Guardar historial
            self.memory.append({"role": "user", "content": query})
            self.memory.append({"role": "assistant", "content": response})

            return response

        # Segundo intento: Fallback usando metadatos
        metadata_keywords = {
            "autor": "author_real",
            "autores": "author_real",
            "quién escribió": "author_real",
            "año": "year",
            "fecha": "year",
            "publicación": "year",
            "doi": "doi",
        }

        for keyword, field in metadata_keywords.items():
            if keyword in q:
                values = [d.metadata.get(field) for d in self.documents if d.metadata.get(field)]
                if values:
                    unique_values = set(str(v) for v in values)
                    return ", ".join(unique_values)

        # Fallo completo
        return "No encontrado en el documento."
