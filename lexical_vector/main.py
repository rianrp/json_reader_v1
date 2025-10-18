"""
Módulo de busca léxica e vetorial
Responsável por:
- search_lexical(query)
- search_vector(query_embedding) 
- hybrid_merge(lexical_results, vector_results)
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import faiss
from whoosh import index as whoosh_index
from whoosh.qparser import QueryParser, MultifieldPlugin
from whoosh.filedb.filestore import FileStorage
from sentence_transformers import SentenceTransformer
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


@dataclass
class SearchResult:
    """Classe para representar um resultado de busca"""
    doc_id: str
    title: str
    content: str
    score: float
    metadata: Optional[str] = None
    method: str = "unknown"  # lexical, vector, hybrid


class LexicalSearcher:
    """Classe para busca léxica usando Whoosh"""
    
    def __init__(self):
        self.index_dir = config.WHOOSH_INDEX_DIR
        self.index = None
        self._load_index()
    
    def _load_index(self):
        """Carrega o índice Whoosh"""
        try:
            storage = FileStorage(str(self.index_dir))
            self.index = storage.open_index(indexname="lexical")
            logger.info("Índice léxico carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar índice léxico: {e}")
            self.index = None
    
    def search_lexical(self, query: str, top_k: int = None) -> List[SearchResult]:
        """
        Realiza busca léxica
        
        Args:
            query: Consulta em texto
            top_k: Número máximo de resultados
            
        Returns:
            Lista de SearchResult ordenados por relevância
        """
        if top_k is None:
            top_k = config.TOP_K_LEXICAL
        
        if not self.index:
            logger.error("Índice léxico não disponível")
            return []
        
        try:
            with self.index.searcher() as searcher:
                # Criar parser para busca em múltiplos campos
                parser = QueryParser("content", self.index.schema)
                parser.add_plugin(MultifieldPlugin(["title", "content"]))
                
                # Parse da query
                parsed_query = parser.parse(query)
                
                # Executar busca
                results = searcher.search(parsed_query, limit=top_k)
                
                # Converter para SearchResult
                search_results = []
                for hit in results:
                    result = SearchResult(
                        doc_id=hit['id'],
                        title=hit['title'],
                        content=hit['content'],
                        score=hit.score,
                        metadata=hit.get('metadata'),
                        method="lexical"
                    )
                    search_results.append(result)
                
                logger.info(f"Busca léxica retornou {len(search_results)} resultados")
                return search_results
                
        except Exception as e:
            logger.error(f"Erro na busca léxica: {e}")
            return []


class VectorSearcher:
    """Classe para busca vetorial usando FAISS"""
    
    def __init__(self):
        self.index_path = config.FAISS_INDEX_PATH
        self.model_name = config.SENTENCE_TRANSFORMER_MODEL
        self.model = SentenceTransformer(self.model_name)
        
        self.faiss_index = None
        self.doc_ids = []
        self.documents_cache = {}
        
        self._load_index()
        self._load_documents_cache()
    
    def _load_index(self):
        """Carrega o índice FAISS"""
        try:
            self.faiss_index = faiss.read_index(str(self.index_path))
            
            # Carregar mapeamento de IDs
            id_mapping_path = self.index_path.with_suffix('.ids')
            with open(id_mapping_path, 'r') as f:
                self.doc_ids = json.load(f)
            
            logger.info("Índice vetorial carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar índice vetorial: {e}")
            self.faiss_index = None
    
    def _load_documents_cache(self):
        """Carrega documentos em cache para acesso rápido"""
        try:
            import sqlite3
            with sqlite3.connect(config.SQLITE_DB_PATH) as conn:
                cursor = conn.execute("SELECT id, title, content, metadata FROM documents")
                for row in cursor.fetchall():
                    self.documents_cache[row[0]] = {
                        'title': row[1],
                        'content': row[2],
                        'metadata': row[3]
                    }
        except Exception as e:
            logger.error(f"Erro ao carregar cache de documentos: {e}")
    
    def search_vector(self, query: str, top_k: int = None) -> List[SearchResult]:
        """
        Realiza busca vetorial
        
        Args:
            query: Consulta em texto
            top_k: Número máximo de resultados
            
        Returns:
            Lista de SearchResult ordenados por similaridade
        """
        if top_k is None:
            top_k = config.TOP_K_VECTOR
        
        if not self.faiss_index:
            logger.error("Índice vetorial não disponível")
            return []
        
        try:
            # Gerar embedding da query
            query_embedding = self.model.encode([query])
            
            # Normalizar para cosine similarity
            faiss.normalize_L2(query_embedding.astype(np.float32))
            
            # Buscar no índice FAISS
            scores, indices = self.faiss_index.search(
                query_embedding.astype(np.float32), 
                top_k
            )
            
            # Converter para SearchResult
            search_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.doc_ids):
                    doc_id = self.doc_ids[idx]
                    
                    if doc_id in self.documents_cache:
                        doc_data = self.documents_cache[doc_id]
                        result = SearchResult(
                            doc_id=doc_id,
                            title=doc_data['title'],
                            content=doc_data['content'],
                            score=float(score),
                            metadata=doc_data['metadata'],
                            method="vector"
                        )
                        search_results.append(result)
            
            logger.info(f"Busca vetorial retornou {len(search_results)} resultados")
            return search_results
            
        except Exception as e:
            logger.error(f"Erro na busca vetorial: {e}")
            return []


class HybridSearcher:
    """Classe para busca híbrida combinando resultados léxicos e vetoriais"""
    
    def __init__(self):
        self.lexical_searcher = LexicalSearcher()
        self.vector_searcher = VectorSearcher()
    
    def hybrid_merge(
        self, 
        lexical_results: List[SearchResult], 
        vector_results: List[SearchResult],
        lexical_weight: float = None,
        vector_weight: float = None
    ) -> List[SearchResult]:
        """
        Combina resultados léxicos e vetoriais
        
        Args:
            lexical_results: Resultados da busca léxica
            vector_results: Resultados da busca vetorial
            lexical_weight: Peso para resultados léxicos
            vector_weight: Peso para resultados vetoriais
            
        Returns:
            Lista combinada e re-rankeada de SearchResult
        """
        if lexical_weight is None:
            lexical_weight = config.LEXICAL_WEIGHT
        if vector_weight is None:
            vector_weight = config.VECTOR_WEIGHT
        
        # Normalizar pesos
        total_weight = lexical_weight + vector_weight
        lexical_weight = lexical_weight / total_weight
        vector_weight = vector_weight / total_weight
        
        # Criar dicionário para combinar resultados
        combined_results = {}
        
        # Normalizar scores para [0, 1]
        if lexical_results:
            max_lexical_score = max(r.score for r in lexical_results)
            min_lexical_score = min(r.score for r in lexical_results)
            lexical_range = max_lexical_score - min_lexical_score or 1
        
        if vector_results:
            max_vector_score = max(r.score for r in vector_results)
            min_vector_score = min(r.score for r in vector_results)
            vector_range = max_vector_score - min_vector_score or 1
        
        # Processar resultados léxicos
        for result in lexical_results:
            normalized_score = (result.score - min_lexical_score) / lexical_range
            weighted_score = normalized_score * lexical_weight
            
            if result.doc_id in combined_results:
                combined_results[result.doc_id].score += weighted_score
            else:
                new_result = SearchResult(
                    doc_id=result.doc_id,
                    title=result.title,
                    content=result.content,
                    score=weighted_score,
                    metadata=result.metadata,
                    method="hybrid"
                )
                combined_results[result.doc_id] = new_result
        
        # Processar resultados vetoriais
        for result in vector_results:
            normalized_score = (result.score - min_vector_score) / vector_range
            weighted_score = normalized_score * vector_weight
            
            if result.doc_id in combined_results:
                combined_results[result.doc_id].score += weighted_score
            else:
                new_result = SearchResult(
                    doc_id=result.doc_id,
                    title=result.title,
                    content=result.content,
                    score=weighted_score,
                    metadata=result.metadata,
                    method="hybrid"
                )
                combined_results[result.doc_id] = new_result
        
        # Ordenar por score combinado
        final_results = sorted(
            combined_results.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        logger.info(f"Busca híbrida combinou {len(final_results)} resultados únicos")
        return final_results
    
    def search(
        self, 
        query: str, 
        top_k: int = None,
        use_lexical: bool = True,
        use_vector: bool = True
    ) -> List[SearchResult]:
        """
        Realiza busca híbrida completa
        
        Args:
            query: Consulta em texto
            top_k: Número máximo de resultados finais
            use_lexical: Se deve usar busca léxica
            use_vector: Se deve usar busca vetorial
            
        Returns:
            Lista final de resultados híbridos
        """
        if top_k is None:
            top_k = config.TOP_K_FINAL
        
        lexical_results = []
        vector_results = []
        
        # Executar busca léxica
        if use_lexical:
            lexical_results = self.lexical_searcher.search_lexical(query)
        
        # Executar busca vetorial
        if use_vector:
            vector_results = self.vector_searcher.search_vector(query)
        
        # Se apenas um método foi usado, retornar diretamente
        if not use_lexical and use_vector:
            return vector_results[:top_k]
        elif use_lexical and not use_vector:
            return lexical_results[:top_k]
        
        # Combinar resultados
        hybrid_results = self.hybrid_merge(lexical_results, vector_results)
        
        return hybrid_results[:top_k]


def main():
    """Função principal para teste"""
    
    # Criar instância do buscador híbrido
    searcher = HybridSearcher()
    
    # Teste de busca
    query = "Python programming"
    
    print(f"🔍 Buscando: '{query}'")
    print("-" * 50)
    
    # Busca léxica apenas
    print("📚 Busca Léxica:")
    lexical_results = searcher.lexical_searcher.search_lexical(query, top_k=5)
    for i, result in enumerate(lexical_results, 1):
        print(f"{i}. {result.title} (Score: {result.score:.3f})")
    
    print()
    
    # Busca vetorial apenas  
    print("🧠 Busca Vetorial:")
    vector_results = searcher.vector_searcher.search_vector(query, top_k=5)
    for i, result in enumerate(vector_results, 1):
        print(f"{i}. {result.title} (Score: {result.score:.3f})")
    
    print()
    
    # Busca híbrida
    print("🔀 Busca Híbrida:")
    hybrid_results = searcher.search(query, top_k=5)
    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. {result.title} (Score: {result.score:.3f})")


if __name__ == "__main__":
    main()
