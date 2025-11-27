from typing import List, Optional, Dict
import os
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from pypdf import PdfReader  # nota: pip install pypdf si se usa en otro ambiente descargar este paquete para no egenrar error

from src.config import RAGConfig


class DocumentProcessor:
    """Handle document loading, metadata extraction and processing."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

    def load_documents(self, file_path: str) -> List[Document]:
        """
        1. Carga el PDF (1 Document por página)
        2. Extrae metadata (autor/título PDF + autor inferido)
        3. Intenta extraer el abstract
        4. Genera tags simples
        5. Inyecta todo eso en metadata
        6. Parte en chunks
        7. Genera un TXT de verificación con TEXTO + METADATA
        """
        loader = PyPDFLoader(file_path)
        documents: List[Document] = loader.load()

        if not documents:
            return []

        # 2) Metadata básica del PDF + inferida
        pdf_metadata = self._extract_pdf_metadata(file_path)

        # 3) Abstract en primeras páginas
        full_text_first_pages = "\n\n".join(
            doc.page_content for doc in documents[:3]
        )
        abstract = self._extract_abstract(full_text_first_pages)

        # 4) Tags
        tags = self._generate_tags(
            title=pdf_metadata.get("title"),
            abstract=abstract
        )

        # 5) Meter metadata en cada página
        for i, d in enumerate(documents):
            d.metadata.update(pdf_metadata)
            d.metadata["abstract"] = abstract
            d.metadata["tags"] = tags

            # page viene 0-based; lo convertimos a 1-based
            page_idx = d.metadata.get("page", i)
            d.metadata["page_number"] = page_idx + 1
            d.metadata["doc_id"] = file_path

        # 6) Chunks
        chunked_docs: List[Document] = self.text_splitter.split_documents(documents)

        # 7) TXT de verificación
        dir_name = os.path.dirname(file_path) or "."
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_txt_path = os.path.join(dir_name, f"{base_name}_VERIFICACION.txt")

        with open(output_txt_path, "w", encoding="utf-8") as f:
            for idx, doc in enumerate(chunked_docs):
                f.write(f"================= CHUNK {idx} =================\n")
                f.write(">>> TEXTO DEL CHUNK:\n")
                f.write(doc.page_content)
                f.write("\n\n")
                f.write(">>> METADATA DEL CHUNK:\n")
                for key, value in doc.metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n\n-----------------------------------------\n\n")

        print(f" Archivo de verificación creado: {output_txt_path}\n")

        return chunked_docs

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize and return the embedding model."""
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': self.config.device},
            encode_kwargs={'normalize_embeddings': False}
        )

    # -------- METADATA PDF --------
    def _extract_pdf_metadata(self, file_path: str) -> Dict:
        """
        Devuelve un diccionario con:
        - author_pdf  : lo que dice la metadata del PDF (si existe)
        - title_pdf   : lo que dice la metadata del PDF (si existe)
        - author_real : autores inferidos del texto de la primera página (si se detectan)
        - author      : atajo -> author_real si existe, si no author_pdf
        - title       : por ahora usamos title_pdf (se podría inferir título real después)
        - year        : año inferido de la fecha de creación (si existe)
        - source      : ruta del archivo
        """
        reader = PdfReader(file_path)
        meta = reader.metadata or {}

        # Lo que dice el PDF en su metadata cruda
        author_meta = getattr(meta, "author", None) or meta.get("/Author")
        title_meta = getattr(meta, "title", None) or meta.get("/Title")

        # Texto de la primera página (para inferir autores reales)
        try:
            first_page_text = reader.pages[0].extract_text() or ""
        except Exception:
            first_page_text = ""

        # Autor inferido del contenido
        author_guessed = self._guess_authors_from_first_page(first_page_text)

        # Campos "real" (desde el contenido)
        author_real = author_guessed

        # Campo de conveniencia: author el real (contenido) y si no, el del PDF
        author_final = author_real or author_meta
        title_final = title_meta  

        # Año desde la fecha de creación
        creation_date = (
            meta.get("/CreationDate")
            or meta.get("creationdate")
            or ""
        )
        year = None
        if isinstance(creation_date, str):
            match = re.search(r"(19|20)\d{2}", creation_date)
            if match:
                year = int(match.group(0))

        return {
            "source": file_path,

            # Campos "crudos" del PDF
            "author_pdf": author_meta,
            "title_pdf": title_meta,

            # Campos inferidos del contenido
            "author_real": author_real,

            # Campos de conveniencia (para filtros genéricos)
            "author": author_final,
            "title": title_final,

            "year": year,
        }

    def _guess_authors_from_first_page(self, text: str) -> Optional[str]:
        """
        Heurística para detectar autores en la primera página:
        - Busca patrones de nombres propios (2–4 palabras con mayúscula inicial).
        - Ignora líneas largas (títulos) o con palabras de afiliación / revista.
        """
        if not text:
            return None

        # Separar en líneas limpias
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # Palabras que NO queremos en la línea de autores
        bad_keywords = [
            "universidad", "facultad", "coordinación", "división",
            "cd. mx", "méxico", "cuaieed", "unam", "investigación",
            "revista", "vol.", "vol ", "núm", "número", "issn", "julio",
            "septiembre", "departamento", "dirección general", "acervos digitales"
        ]

        # Patrón de nombre propio: 2–4 palabras con mayúscula inicial usando re para identificar normbres
        name_pattern = re.compile(
            r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?: [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3}"
        )

        candidates = []

        for line in lines:
            lower = line.lower()

            # Ignorar líneas muy largas (probablemente títulos en mayúsculas)
            if len(line) > 80:
                continue

            # Ignorar líneas con palabras extrañas xd
            if any(bad in lower for bad in bad_keywords):
                continue

            # Buscar nombres en la línea
            matches = name_pattern.findall(line)
            if matches:
                # Guardamos todos los matches de la línea
                candidates.extend(matches)

        # Si encontramos candidatos, devolvemos únicos en orden
        if candidates:
            unique = list(dict.fromkeys(candidates))  # elimina duplicados manteniendo orden
            return ", ".join(unique)

        return None

    # -------- ABSTRACT --------
    def _extract_abstract(self, text: str) -> Optional[str]:
        if not text:
            return None

        lower = text.lower()
        header_candidates = ["resumen", "abstract", "summary"]
        end_markers = ["palabras clave", "keywords"]

        for header in header_candidates:
            idx = lower.find(header)
            if idx == -1:
                continue

            start = idx + len(header)

            ends = []
            for marker in end_markers:
                j = lower.find(marker, start)
                if j != -1:
                    ends.append(j)

            if ends:
                end = min(ends)
            else:
                end = min(len(text), start + 2000)

            abstract_text = text[start:end].strip()
            if abstract_text:
                return abstract_text

        return None

    # -------- TAGS --------
    def _generate_tags(self, title: Optional[str], abstract: Optional[str]) -> List[str]:
        text = ((title or "") + " " + (abstract or "")).lower()
        tags: List[str] = []

        if "inteligencia artificial" in text or "artificial intelligence" in text:
            tags.append("ai")

        if any(word in text for word in [
            "aprendizaje automático", "machine learning", "aprendizaje de máquina"
        ]):
            tags.append("machine-learning")

        if any(word in text for word in [
            "aprendizaje profundo", "deep learning"
        ]):
            tags.append("deep-learning")

        if any(word in text for word in [
            "procesamiento del lenguaje natural", "pln", "nlp", "natural language processing"
        ]):
            tags.append("nlp")

        if any(word in text for word in [
            "retrieval augmented generation", "rag"
        ]):
            tags.append("rag")

        if any(word in text for word in [
            "revisión de la literatura", "literature review", "systematic review"
        ]):
            tags.append("literature-review")

        if not tags and (title or abstract):
            tags.append("general-paper")

        return tags
