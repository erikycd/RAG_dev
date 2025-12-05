import os
import numpy as np
from neo4j import GraphDatabase, Driver
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RAGConfig

# Nota: Neo4j Python Driver debe estar instalado (pip install neo4j)

class Neo4jGraphIndexer:
    """
    Indexador que guarda los chunks de texto y sus embeddings en Neo4j.
    - Crea nodos (:Chunk)
    - Crea aristas (:SIMILAR_TO) basado en la similitud de los embeddings.
    """

    def __init__(self, config: RAGConfig, embedder):
        self.config = config
        self.embedder = embedder
        
        # Conexión a Neo4j (usa las variables de entorno de tu archivo .env)
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "neo4jpassword")
        
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self._check_connection()
        self._setup_constraints()
        self._setup_vector_index()

    def _check_connection(self):
        try:
            self.driver.verify_connectivity()
            print(" Conexión a Neo4j exitosa.")
        except Exception as e:
            print(f" Error de conexión a Neo4j: {e}")
            print(" Asegúrate de que el contenedor de Docker 'neo4j_graph' esté corriendo.")
            raise

    def _setup_constraints(self):
        """Crea una restricción para asegurar unicidad de IDs (Neo4j 5.x)."""
        query = """
        CREATE CONSTRAINT chunk_id_constraint IF NOT EXISTS
        FOR (c:Chunk)
        REQUIRE c.chunk_id IS UNIQUE
        """
        with self.driver.session() as session:
            session.run(query)

    def _setup_vector_index(self):
        """Crea el índice vectorial para búsquedas por embedding."""
        query = f"""
        CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
        FOR (c:Chunk) ON (c.embedding) 
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {len(self.embedder.embed_query("A"))},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        with self.driver.session() as session:
            session.run(query)

    def index_documents(self, chunk_docs: List[dict]):
        """
        Procesa e indexa una lista de chunks.
        """
        if not chunk_docs:
            print(" No hay chunks para indexar.")
            return

        print(f" Indexando {len(chunk_docs)} chunks en Neo4j...")
        
        # 1. Crear nodos y obtener embeddings
        chunks_to_add = []
        texts = [d.page_content for d in chunk_docs]
        embeddings = self.embedder.embed_documents(texts) # batch embedding
        
        for i, doc in enumerate(chunk_docs):
            # Crear un ID único para el chunk (Source + Página + Índice del chunk)
            chunk_id = f"{os.path.basename(doc.metadata.get('doc_id',''))}::p{doc.metadata.get('page_number',0)}::c{i}"
            
            chunks_to_add.append({
                "chunk_id": chunk_id,
                "text": doc.page_content,
                "embedding": embeddings[i],
                # Copiar metadata como propiedades separadas en el nodo
                **doc.metadata
            })
        
        # Transacción para crear NODOS (:Chunk)
        self._create_chunk_nodes(chunks_to_add)
        
        # 2. Crear relaciones de Similitud (:SIMILAR_TO)
        self._create_similarity_relationships(chunks_to_add, embeddings)
        
        print(f" Indexación en Neo4j completada. Nodos: {len(chunks_to_add)}.")


    def _create_chunk_nodes(self, chunks: List[Dict]):
        """Crea los nodos :Chunk en Neo4j."""
        query = """
        UNWIND $chunks AS chunk
        MERGE (c:Chunk {chunk_id: chunk.chunk_id})
        SET c += chunk
        """
        with self.driver.session() as session:
            session.run(query, chunks=chunks)

    def _create_similarity_relationships(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Crea aristas de similitud entre los nuevos chunks."""
        
        # Convertir embeddings a numpy array para cálculo rápido
        vecs = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        
        # Calcular similitud coseno entre todos los chunks añadidos
        sim = cosine_similarity(vecs)
        n = sim.shape[0]
        
        relationships = []
        
        for i in range(n):
            # Obtener índices de los chunks más similares (incluye a sí mismo)
            idxs = np.argsort(-sim[i])
            
            # Recorrer los top-K o sobre el umbral
            # Empezamos desde el índice 1 para saltar la similitud con sigo mismo
            for j in idxs[1:]: 
                score = float(sim[i, j])
                
                # Aplicar las restricciones de configuración
                if score >= self.config.edge_similarity_threshold or (j < self.config.edge_top_k + 1):
                    rel = {
                        "chunk_id_1": chunks[i]["chunk_id"],
                        "chunk_id_2": chunks[j]["chunk_id"],
                        "score": score
                    }
                    relationships.append(rel)
                else:
                    break # Optimización: si ya no pasa el umbral, el resto tampoco

        # Transacción para crear RELACIONES (:SIMILAR_TO)
        rel_query = """
        UNWIND $rels AS rel
        MATCH (c1:Chunk {chunk_id: rel.chunk_id_1})
        MATCH (c2:Chunk {chunk_id: rel.chunk_id_2})
        MERGE (c1)-[s:SIMILAR_TO]-(c2)
        SET s.weight = rel.score
        """
        with self.driver.session() as session:
            session.run(rel_query, rels=relationships)
    def clear_graph(self):
        """Remove all chunks and similarity relationships."""
        query = """
        MATCH (c:Chunk)
        DETACH DELETE c
        """
        with self.driver.session() as session:
            session.run(query)
        print("Neo4j graph cleaned.")


    def close(self):
        """Cierra la conexión al driver de Neo4j."""
        self.driver.close()
