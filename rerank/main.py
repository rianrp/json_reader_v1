"""
Módulo de reranqueamento
Responsável por:
- Reordenar a lista com base na relevância
- BM25 + cosine + heurística + ML opcional
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
import json

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


@dataclass
class RerankResult:
    """Resultado com score de reranking"""
    doc_id: str
    title: str
    content: str
    original_score: float
    bm25_score: float = 0.0
    cosine_score: float = 0.0
    heuristic_score: float = 0.0
    final_score: float = 0.0
    metadata: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)


class BM25Reranker:
    """Reranking usando BM25"""
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
    
    def fit(self, documents: List[Dict[str, Any]]):
        """
        Treina o modelo BM25 com os documentos
        
        Args:
            documents: Lista de documentos com 'id', 'title', 'content'
        """
        try:
            # Preparar corpus para BM25
            corpus = []
            self.doc_ids = []
            
            for doc in documents:
                # Combinar título e conteúdo
                text = f"{doc['title']} {doc['content']}"
                # Tokenização simples
                tokens = text.lower().split()
                corpus.append(tokens)
                self.doc_ids.append(doc['id'])
            
            # Criar modelo BM25
            self.bm25 = BM25Okapi(corpus)
            self.documents = documents
            
            logger.info(f"BM25 treinado com {len(documents)} documentos")
            
        except Exception as e:
            logger.error(f"Erro ao treinar BM25: {e}")
    
    def score_documents(self, query: str, doc_ids: List[str]) -> Dict[str, float]:
        """
        Calcula scores BM25 para documentos específicos
        
        Args:
            query: Query de busca
            doc_ids: IDs dos documentos a serem pontuados
            
        Returns:
            Dict mapeando doc_id para score BM25
        """
        if not self.bm25:
            logger.error("BM25 não foi treinado")
            return {}
        
        try:
            # Tokenizar query
            query_tokens = query.lower().split()
            
            # Calcular scores para todos os documentos
            all_scores = self.bm25.get_scores(query_tokens)
            
            # Mapear scores para doc_ids específicos
            scores = {}
            for doc_id in doc_ids:
                if doc_id in self.doc_ids:
                    idx = self.doc_ids.index(doc_id)
                    scores[doc_id] = float(all_scores[idx])
                else:
                    scores[doc_id] = 0.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Erro ao calcular scores BM25: {e}")
            return {doc_id: 0.0 for doc_id in doc_ids}


class CosineReranker:
    """Reranking usando similaridade de cosseno"""
    
    def __init__(self):
        self.model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
    
    def score_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calcula similaridade de cosseno entre query e documentos
        
        Args:
            query: Query de busca
            documents: Lista de documentos com 'id', 'title', 'content'
            
        Returns:
            Dict mapeando doc_id para score de cosseno
        """
        try:
            # Preparar textos
            texts = [query]  # Query primeiro
            doc_ids = []
            
            for doc in documents:
                text = f"{doc['title']}\n{doc['content']}"
                texts.append(text)
                doc_ids.append(doc['id'])
            
            # Gerar embeddings
            embeddings = self.model.encode(texts)
            
            # Calcular similaridade com a query (primeiro embedding)
            query_embedding = embeddings[0:1]
            doc_embeddings = embeddings[1:]
            
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Mapear para doc_ids
            scores = {}
            for doc_id, similarity in zip(doc_ids, similarities):
                scores[doc_id] = float(similarity)
            
            return scores
            
        except Exception as e:
            logger.error(f"Erro ao calcular similaridade de cosseno: {e}")
            return {doc['id']: 0.0 for doc in documents}


class HeuristicReranker:
    """Reranking usando heurísticas"""
    
    def score_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calcula scores heurísticos
        
        Args:
            query: Query de busca
            documents: Lista de documentos
            
        Returns:
            Dict mapeando doc_id para score heurístico
        """
        try:
            query_lower = query.lower()
            query_tokens = set(query_lower.split())
            
            scores = {}
            
            for doc in documents:
                score = 0.0
                
                title_lower = doc['title'].lower()
                content_lower = doc['content'].lower()
                
                # 1. Matches exatos no título (peso alto)
                if query_lower in title_lower:
                    score += 2.0
                
                # 2. Matches exatos no conteúdo
                if query_lower in content_lower:
                    score += 1.0
                
                # 3. Número de tokens da query encontrados
                title_tokens = set(title_lower.split())
                content_tokens = set(content_lower.split())
                
                title_matches = len(query_tokens.intersection(title_tokens))
                content_matches = len(query_tokens.intersection(content_tokens))
                
                if len(query_tokens) > 0:
                    # Proporção de tokens encontrados no título
                    score += (title_matches / len(query_tokens)) * 1.5
                    # Proporção de tokens encontrados no conteúdo  
                    score += (content_matches / len(query_tokens)) * 0.5
                
                # 4. Penalizar documentos muito longos ou muito curtos
                content_length = len(doc['content'])
                if 100 <= content_length <= 2000:  # Tamanho ideal
                    score += 0.2
                elif content_length < 50:  # Muito curto
                    score -= 0.3
                elif content_length > 5000:  # Muito longo
                    score -= 0.2
                
                # 5. Bonus para documentos com títulos informativos
                title_length = len(doc['title'])
                if 10 <= title_length <= 100:
                    score += 0.1
                
                scores[doc['id']] = score
            
            return scores
            
        except Exception as e:
            logger.error(f"Erro ao calcular scores heurísticos: {e}")
            return {doc['id']: 0.0 for doc in documents}


class MLReranker:
    """Reranking usando ML (opcional/placeholder)"""
    
    def __init__(self):
        self.model = None
        self.features_scaler = None
    
    def extract_features(
        self, 
        query: str, 
        document: Dict[str, Any],
        bm25_score: float,
        cosine_score: float,
        heuristic_score: float
    ) -> np.ndarray:
        """
        Extrai features para ML
        
        Args:
            query: Query de busca
            document: Documento
            bm25_score: Score BM25
            cosine_score: Score de cosseno
            heuristic_score: Score heurístico
            
        Returns:
            Array de features
        """
        try:
            # Features básicas
            features = []
            
            # Scores de outros métodos
            features.extend([bm25_score, cosine_score, heuristic_score])
            
            # Features de texto
            query_lower = query.lower()
            title_lower = document['title'].lower()
            content_lower = document['content'].lower()
            
            # Lengths
            features.append(len(query.split()))
            features.append(len(document['title'].split()))
            features.append(len(document['content'].split()))
            
            # Exact matches
            features.append(1.0 if query_lower in title_lower else 0.0)
            features.append(1.0 if query_lower in content_lower else 0.0)
            
            # Token overlap
            query_tokens = set(query_lower.split())
            title_tokens = set(title_lower.split())
            content_tokens = set(content_lower.split())
            
            if len(query_tokens) > 0:
                features.append(len(query_tokens.intersection(title_tokens)) / len(query_tokens))
                features.append(len(query_tokens.intersection(content_tokens)) / len(query_tokens))
            else:
                features.extend([0.0, 0.0])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Erro ao extrair features: {e}")
            return np.zeros(10)  # Features dummy
    
    def score_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]],
        bm25_scores: Dict[str, float],
        cosine_scores: Dict[str, float],
        heuristic_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Scores usando ML (implementação básica)
        
        Returns:
            Dict mapeando doc_id para score ML
        """
        # Por enquanto, apenas combina os scores existentes
        # Em uma implementação real, você treinaria um modelo ML
        
        scores = {}
        
        for doc in documents:
            doc_id = doc['id']
            
            bm25 = bm25_scores.get(doc_id, 0.0)
            cosine = cosine_scores.get(doc_id, 0.0)
            heuristic = heuristic_scores.get(doc_id, 0.0)
            
            # Combinação simples ponderada
            ml_score = (bm25 * 0.3) + (cosine * 0.4) + (heuristic * 0.3)
            scores[doc_id] = ml_score
        
        return scores


class Reranker:
    """Classe principal de reranking"""
    
    def __init__(self):
        self.bm25_reranker = BM25Reranker()
        self.cosine_reranker = CosineReranker()
        self.heuristic_reranker = HeuristicReranker()
        self.ml_reranker = MLReranker()
        
        self.weights = {
            'bm25': 0.25,
            'cosine': 0.35,
            'heuristic': 0.25,
            'ml': 0.15
        }
    
    def fit_bm25(self, documents: List[Dict[str, Any]]):
        """Treina o componente BM25"""
        self.bm25_reranker.fit(documents)
    
    def rerank(
        self, 
        query: str, 
        search_results: List[Any],  # Pode ser SearchResult ou dict
        top_k: int = None
    ) -> List[RerankResult]:
        """
        Rerank dos resultados de busca
        
        Args:
            query: Query original
            search_results: Resultados de busca
            top_k: Número de resultados finais
            
        Returns:
            Lista de RerankResult ordenada por score final
        """
        if top_k is None:
            top_k = config.TOP_K_FINAL
        
        if not search_results:
            return []
        
        try:
            # Converter search_results para formato padrão
            documents = []
            for result in search_results:
                if hasattr(result, 'doc_id'):  # SearchResult
                    doc = {
                        'id': result.doc_id,
                        'title': result.title,
                        'content': result.content,
                        'original_score': result.score
                    }
                else:  # Dict
                    doc = {
                        'id': result['id'],
                        'title': result['title'],
                        'content': result['content'],
                        'original_score': result.get('score', 0.0)
                    }
                documents.append(doc)
            
            # Calcular scores de diferentes métodos
            bm25_scores = self.bm25_reranker.score_documents(
                query, [doc['id'] for doc in documents]
            )
            cosine_scores = self.cosine_reranker.score_documents(query, documents)
            heuristic_scores = self.heuristic_reranker.score_documents(query, documents)
            ml_scores = self.ml_reranker.score_documents(
                query, documents, bm25_scores, cosine_scores, heuristic_scores
            )
            
            # Normalizar scores
            def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
                if not scores:
                    return scores
                values = list(scores.values())
                if not values or max(values) == min(values):
                    return {k: 0.0 for k in scores.keys()}
                
                min_val, max_val = min(values), max(values)
                return {
                    k: (v - min_val) / (max_val - min_val) 
                    for k, v in scores.items()
                }
            
            bm25_norm = normalize_scores(bm25_scores)
            cosine_norm = normalize_scores(cosine_scores)
            heuristic_norm = normalize_scores(heuristic_scores)
            ml_norm = normalize_scores(ml_scores)
            
            # Combinar scores
            rerank_results = []
            
            for doc in documents:
                doc_id = doc['id']
                
                bm25_score = bm25_norm.get(doc_id, 0.0)
                cosine_score = cosine_norm.get(doc_id, 0.0)
                heuristic_score = heuristic_norm.get(doc_id, 0.0)
                ml_score = ml_norm.get(doc_id, 0.0)
                
                # Score final ponderado
                final_score = (
                    bm25_score * self.weights['bm25'] +
                    cosine_score * self.weights['cosine'] +
                    heuristic_score * self.weights['heuristic'] +
                    ml_score * self.weights['ml']
                )
                
                result = RerankResult(
                    doc_id=doc_id,
                    title=doc['title'],
                    content=doc['content'],
                    original_score=doc['original_score'],
                    bm25_score=bm25_score,
                    cosine_score=cosine_score,
                    heuristic_score=heuristic_score,
                    final_score=final_score,
                    features={
                        'bm25': bm25_score,
                        'cosine': cosine_score,
                        'heuristic': heuristic_score,
                        'ml': ml_score
                    }
                )
                rerank_results.append(result)
            
            # Ordenar por score final
            rerank_results.sort(key=lambda x: x.final_score, reverse=True)
            
            logger.info(f"Rerank completo: {len(rerank_results)} documentos processados")
            return rerank_results[:top_k]
            
        except Exception as e:
            logger.error(f"Erro durante reranking: {e}")
            return []


def main():
    """Função principal para teste"""
    
    # Exemplo de documentos
    documents = [
        {
            'id': '1',
            'title': 'Python Programming Guide',
            'content': 'Python is a high-level programming language...',
            'score': 0.8
        },
        {
            'id': '2', 
            'title': 'Machine Learning with Python',
            'content': 'Learn machine learning using Python libraries...',
            'score': 0.7
        }
    ]
    
    # Criar reranker e treinar BM25
    reranker = Reranker()
    reranker.fit_bm25(documents)
    
    # Teste de reranking
    query = "Python programming"
    
    print(f"🔄 Reranking para: '{query}'")
    print("-" * 50)
    
    results = reranker.rerank(query, documents, top_k=5)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.title}")
        print(f"   Score Original: {result.original_score:.3f}")
        print(f"   BM25: {result.bm25_score:.3f}")
        print(f"   Cosine: {result.cosine_score:.3f}")
        print(f"   Heuristic: {result.heuristic_score:.3f}")
        print(f"   Final: {result.final_score:.3f}")
        print()


if __name__ == "__main__":
    main()
