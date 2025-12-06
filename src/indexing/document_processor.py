from typing import List, Optional, Dict
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

from pypdf import PdfReader
from src.config import RAGConfig

# -------- Regex para extraer información estructurada --------
DOI_REGEX = r"10\.\d{4,9}\/[-._;()\/:A-Za-z0-9]+"
ISSN_REGEX = r"\d{4}-\d{3}[\dX]"
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
ORCID_REGEX = r"https?:\/\/orcid\.org\/[\d\-]{15,}"

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
        loader = PyMuPDFLoader(file_path)
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

        # --- INYECCIÓN DE METADATA AL TEXTO ---
        for i, d in enumerate(documents):
            d.metadata.update(pdf_metadata)
            d.metadata["tags"] = tags
            d.metadata["page_number"] = d.metadata.get("page", i) + 1
            d.metadata["doc_id"] = file_path

            section = self._detect_section(d.page_content)
            d.metadata["section"] = section

            # Convertir metadata en encabezado textual para embeddings
            metadata_text = []

            if pdf_metadata.get("title"):
                metadata_text.append(f"Título: {pdf_metadata['title']}")
            if pdf_metadata.get("author_real"):
                metadata_text.append(f"Autor: {pdf_metadata['author_real']}")
            if pdf_metadata.get("year"):
                metadata_text.append(f"Año: {pdf_metadata['year']}")
            if pdf_metadata.get("doi"):
                metadata_text.append(f"DOI: {pdf_metadata['doi']}")

            if metadata_text and i ==0:
                metadata_block = "\n".join(metadata_text) + "\n\n"
                d.page_content = metadata_block + d.page_content

        # 6) Chunking
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
    
    def _extract_pdf_metadata(self, file_path: str) -> Dict:
        reader = PdfReader(file_path)
        meta = reader.metadata or {}

        full_first_page = reader.pages[0].extract_text() or ""

        # DOI
        doi_match = re.search(DOI_REGEX, full_first_page)
        doi = doi_match.group(0) if doi_match else None

        # ORCID autores
        orcids = list(set(re.findall(r"https?:\/\/orcid\.org\/[\d\-]{15,}", full_first_page)))

        # Emails
        emails = re.findall(EMAIL_REGEX, full_first_page)

        # Año desde fechas tipo: Recibido / Aceptado / Publicado
        year_match = re.search(r"(19|20)\d{2}", full_first_page)
        year = int(year_match.group(0)) if year_match else None

        # Extraer autores de forma robusta
        author_real = self._guess_authors_from_first_page(full_first_page)

        # Título real: si hay título en metadata y texto visible
        title = meta.get("/Title") or "Sin título claro"

        return {
            "source": file_path,
            "title": title,
            "author_real": author_real or "Autor no detectado",
            "year": year,
            "doi": doi,
            "emails": emails,
            "orcids": orcids,
            "issn": meta.get("/ISSN"),
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

    def _detect_section(self, text: str) -> str:
        lower = text.lower()

        sections = {
            "resumen": "Resumen",
            "abstract": "Resumen",
            "summary": "Resumen",
            "introducción": "Introducción",
            "introduction": "Introducción",
            "metodología": "Metodología",
            "methods": "Metodología",
            "resultados": "Resultados",
            "results": "Resultados",
            "discusión": "Discusión",
            "discussion": "Discusión",
            "conclusiones": "Conclusiones",
            "conclusion": "Conclusiones",
            "palabras clave": "Palabras clave",
            "keywords": "Palabras clave"
        }

        for key, sec in sections.items():
            if key in lower:
                return sec

        return "Texto general"

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

