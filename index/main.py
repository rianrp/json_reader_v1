"""
Módulo de indexação de documentos
Responsável por:
- Ingestão de documentos
- Criação de índice léxico (Whoosh)
- Criação de índice vetorial (FAISS)
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

import numpy as np
import faiss
from whoosh import fields, index
from whoosh.filedb.filestore import FileStorage
from sentence_transformers import SentenceTransformer
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config, setup_directories


class DocumentIngestion:
    """Classe para ingestão de documentos"""
    
    def __init__(self):
        setup_directories()
        self.db_path = config.SQLITE_DB_PATH
        self._init_database()
    
    def _init_database(self):
        """Inicializa o banco de dados SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def ingest_json_documents(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Ingere documentos de um arquivo JSON
        
        Args:
            json_path: Caminho para o arquivo JSON
            
        Returns:
            Lista de documentos processados
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Se é uma lista de documentos
            if isinstance(data, list):
                documents = data
            # Se é um objeto único
            elif isinstance(data, dict):
                documents = [data]
            else:
                raise ValueError("JSON deve ser uma lista ou objeto")
            
            processed_docs = []
            
            with sqlite3.connect(self.db_path) as conn:
                for doc in documents:
                    doc_id = str(uuid.uuid4())
                    title = doc.get('title', doc.get('name', f'Document {doc_id[:8]}'))
                    content = doc.get('content', doc.get('text', str(doc)))
                    metadata = json.dumps({k: v for k, v in doc.items() if k not in ['title', 'content', 'text']})
                    
                    conn.execute(
                        "INSERT INTO documents (id, title, content, metadata) VALUES (?, ?, ?, ?)",
                        (doc_id, title, content, metadata)
                    )
                    
                    processed_doc = {
                        'id': doc_id,
                        'title': title,
                        'content': content,
                        'metadata': metadata
                    }
                    processed_docs.append(processed_doc)
                
                conn.commit()
            
            logger.info(f"Ingeridos {len(processed_docs)} documentos")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Erro ao ingerir documentos: {e}")
            return []
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retorna todos os documentos do banco"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, title, content, metadata FROM documents")
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'metadata': row[3]
                })
            return documents


class LexicalIndexer:
    """Classe para criação e gerenciamento do índice léxico com Whoosh"""
    
    def __init__(self):
        setup_directories()
        self.index_dir = config.WHOOSH_INDEX_DIR
        self.schema = fields.Schema(
            id=fields.ID(stored=True),
            title=fields.TEXT(stored=True),
            content=fields.TEXT(stored=True),
            metadata=fields.TEXT(stored=True)
        )
    
    def create_index(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Cria o índice léxico
        
        Args:
            documents: Lista de documentos para indexar
            
        Returns:
            True se bem-sucedido
        """
        try:
            # Criar o índice
            storage = FileStorage(str(self.index_dir))
            ix = storage.create_index(self.schema, indexname="lexical")
            
            # Adicionar documentos
            writer = ix.writer()
            for doc in documents:
                writer.add_document(
                    id=doc['id'],
                    title=doc['title'],
                    content=doc['content'],
                    metadata=doc.get('metadata', '')
                )
            writer.commit()
            
            logger.info(f"Índice léxico criado com {len(documents)} documentos")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao criar índice léxico: {e}")
            return False
    
    def get_index(self):
        """Retorna o índice existente"""
        try:
            storage = FileStorage(str(self.index_dir))
            return storage.open_index(indexname="lexical")
        except Exception as e:
            logger.error(f"Erro ao abrir índice: {e}")
            return None


class VectorIndexer:
    """Classe para criação e gerenciamento do índice vetorial com FAISS"""
    
    def __init__(self):
        setup_directories()
        self.index_path = config.FAISS_INDEX_PATH
        self.model_name = config.SENTENCE_TRANSFORMER_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
    def create_index(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Cria o índice vetorial
        
        Args:
            documents: Lista de documentos para indexar
            
        Returns:
            True se bem-sucedido
        """
        try:
            # Preparar textos para embedding
            texts = []
            doc_ids = []
            for doc in documents:
                # Combinar título e conteúdo para melhor representação
                text = f"{doc['title']}\n{doc['content']}"
                texts.append(text)
                doc_ids.append(doc['id'])
            
            # Gerar embeddings
            logger.info("Gerando embeddings...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Criar índice FAISS
            index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine similarity)
            
            # Normalizar embeddings para cosine similarity
            faiss.normalize_L2(embeddings.astype(np.float32))
            
            # Adicionar embeddings ao índice
            index.add(embeddings.astype(np.float32))
            
            # Salvar índice
            faiss.write_index(index, str(self.index_path))
            
            # Salvar mapeamento de IDs
            id_mapping_path = self.index_path.with_suffix('.ids')
            with open(id_mapping_path, 'w') as f:
                json.dump(doc_ids, f)
            
            logger.info(f"Índice vetorial criado com {len(documents)} documentos")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao criar índice vetorial: {e}")
            return False
    
    def load_index(self):
        """Carrega o índice existente"""
        try:
            index = faiss.read_index(str(self.index_path))
            
            id_mapping_path = self.index_path.with_suffix('.ids')
            with open(id_mapping_path, 'r') as f:
                doc_ids = json.load(f)
            
            return index, doc_ids
        except Exception as e:
            logger.error(f"Erro ao carregar índice vetorial: {e}")
            return None, None


class IndexManager:
    """Gerenciador principal dos índices"""
    
    def __init__(self):
        self.ingestion = DocumentIngestion()
        self.lexical_indexer = LexicalIndexer()
        self.vector_indexer = VectorIndexer()
    
    def build_indices_from_json(self, json_path: str) -> bool:
        """
        Constrói todos os índices a partir de um arquivo JSON
        
        Args:
            json_path: Caminho para o arquivo JSON
            
        Returns:
            True se bem-sucedido
        """
        logger.info(f"Iniciando indexação de {json_path}")
        
        # 1. Ingerir documentos
        documents = self.ingestion.ingest_json_documents(json_path)
        if not documents:
            logger.error("Falha na ingestão de documentos")
            return False
        
        # 2. Criar índice léxico
        if not self.lexical_indexer.create_index(documents):
            logger.error("Falha na criação do índice léxico")
            return False
        
        # 3. Criar índice vetorial
        if not self.vector_indexer.create_index(documents):
            logger.error("Falha na criação do índice vetorial")
            return False
        
        logger.info("Indexação completa!")
        return True
    
    def rebuild_indices(self) -> bool:
        """Reconstrói os índices usando documentos existentes no banco"""
        documents = self.ingestion.get_all_documents()
        if not documents:
            logger.warning("Nenhum documento encontrado no banco")
            return False
        
        success = True
        success &= self.lexical_indexer.create_index(documents)
        success &= self.vector_indexer.create_index(documents)
        
        return success


def main():
    """Função principal para teste"""
    manager = IndexManager()
    
    # Exemplo de uso com um arquivo JSON
    sample_json = config.PROJECT_ROOT / "sample_documents.json"
    
    # Criar arquivo de exemplo se não existir
    if not sample_json.exists():
        sample_docs = [
            {
                "title": "Python Programming",
                "content": "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.",
                "category": "programming"
            },
            {
                "title": "Machine Learning Basics",
                "content": "Machine Learning é um subcampo da inteligência artificial que usa algoritmos para aprender padrões.",
                "category": "ai"
            }
        ]
        
        with open(sample_json, 'w', encoding='utf-8') as f:
            json.dump(sample_docs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Arquivo de exemplo criado: {sample_json}")
    
    # Construir índices
    success = manager.build_indices_from_json(str(sample_json))
    if success:
        print("✅ Índices criados com sucesso!")
    else:
        print("❌ Erro na criação dos índices")


if __name__ == "__main__":
    main()
