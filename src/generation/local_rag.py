from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from src.retrieval.rag_model import RAGModel
from src.config import RAGConfig

class LocalRAG(RAGModel):
    """
    RAG-Naive Local LLM con FAISS.
    - Recupera contexto con embeddings desde FAISS
    - Responde usando Local LLMs
    - Mantiene memoria simple para contexto conversacional
    """

    def __init__(self, config: RAGConfig, documents):
        super().__init__(config, documents)
        self.documents = documents
        self.llm = OpenAI(
            base_url = "http://localhost:1234/v1",
            api_key = "lm-studio"
        )
        self.memory = []
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Eres un asistente llamado RAGY.\n\n"
                "Contexto disponible:\n"
                "{context}\n\n"
                "Pregunta:\n"
                "{question}\n\n"
                "Instrucciones:\n"
                "- Trata de responder con información del texto.\n"
                "- Incluye referencias a la página cuando sea posible.\n"
                "- Responde preferentemente en español.\n"
            )
        )

    def generate_response(self, query: str) -> str:
        q = query.lower()

        retrieved_docs = self.retrieve_context(query)

        # Priorizar secciones con contenido relevante
        retrieved_docs_sorted = sorted(
            retrieved_docs,
            key=lambda d: (
                d.metadata.get("section") not in ["Resumen", "Resultados", "Conclusiones"],
                -len(d.page_content)
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
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": formatted_prompt})

            response = self.llm.chat.completions.create(
                model = 'unsloth/deepseek-r1-distill-qwen-7b',
                messages = messages,
                temperature = self.config.temperature
            )

            content = response.choices[0].message.content

            self.memory.append({"role": "user", "content": query})
            self.memory.append({"role": "assistant", "content": content})

            return content

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

        return "No encontrado en el documento."
