"""
Módulo de Telemetria e Aprendizado
Responsável por:
- Log de queries, cliques, sucessos
- Feedback loop (ajustar peso do ranking)
- Análise de desempenho
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import uuid

import pandas as pd
import numpy as np
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config, setup_directories


class EventType(Enum):
    """Tipos de eventos de telemetria"""
    QUERY = "query"
    RESULT_CLICK = "result_click"
    RESULT_RATING = "result_rating"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    ERROR = "error"


class FeedbackType(Enum):
    """Tipos de feedback do usuário"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING_1 = "rating_1"
    RATING_2 = "rating_2" 
    RATING_3 = "rating_3"
    RATING_4 = "rating_4"
    RATING_5 = "rating_5"


@dataclass
class TelemetryEvent:
    """Evento de telemetria"""
    event_id: str
    session_id: str
    event_type: EventType
    timestamp: datetime
    query: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    clicked_result_id: Optional[str] = None
    click_position: Optional[int] = None
    feedback_type: Optional[FeedbackType] = None
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryAnalytics:
    """Análises de uma query"""
    query: str
    total_searches: int
    avg_response_time: float
    click_through_rate: float
    avg_rating: Optional[float]
    success_rate: float
    top_clicked_results: List[Dict[str, Any]]


class TelemetryCollector:
    """Coletor de dados de telemetria"""
    
    def __init__(self):
        setup_directories()
        self.db_path = config.DATA_DIR / "telemetry.db"
        self._init_database()
        self.current_session_id = None
    
    def _init_database(self):
        """Inicializa o banco de dados de telemetria"""
        with sqlite3.connect(self.db_path) as conn:
            # Tabela de eventos
            conn.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    event_type TEXT,
                    timestamp TEXT,
                    query TEXT,
                    results TEXT,
                    clicked_result_id TEXT,
                    click_position INTEGER,
                    feedback_type TEXT,
                    response_time_ms REAL,
                    metadata TEXT
                )
            """)
            
            # Tabela de sessões
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    total_queries INTEGER DEFAULT 0,
                    total_clicks INTEGER DEFAULT 0,
                    avg_response_time REAL,
                    metadata TEXT
                )
            """)
            
            # Tabela de rankings (para aprendizado)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ranking_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    lexical_weight REAL,
                    vector_weight REAL,
                    bm25_weight REAL,
                    cosine_weight REAL,
                    heuristic_weight REAL,
                    performance_score REAL,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    def start_session(self, metadata: Dict[str, Any] = None) -> str:
        """Inicia uma nova sessão"""
        self.current_session_id = str(uuid.uuid4())
        
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            session_id=self.current_session_id,
            event_type=EventType.SESSION_START,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata
        )
        
        self.log_event(event)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, start_time, metadata) VALUES (?, ?, ?)",
                (self.current_session_id, event.timestamp.isoformat(), json.dumps(metadata or {}))
            )
            conn.commit()
        
        logger.info(f"Nova sessão iniciada: {self.current_session_id}")
        return self.current_session_id
    
    def end_session(self, metadata: Dict[str, Any] = None):
        """Finaliza a sessão atual"""
        if not self.current_session_id:
            return
        
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            session_id=self.current_session_id,
            event_type=EventType.SESSION_END,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata
        )
        
        self.log_event(event)
        
        # Atualizar estatísticas da sessão
        self._update_session_stats()
        
        logger.info(f"Sessão finalizada: {self.current_session_id}")
        self.current_session_id = None
    
    def log_query(
        self, 
        query: str, 
        results: List[Dict[str, Any]],
        response_time_ms: float,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Log de uma query"""
        
        if not self.current_session_id:
            self.start_session()
        
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            session_id=self.current_session_id,
            event_type=EventType.QUERY,
            timestamp=datetime.now(timezone.utc),
            query=query,
            results=results,
            response_time_ms=response_time_ms,
            metadata=metadata
        )
        
        self.log_event(event)
        return event.event_id
    
    def log_click(
        self,
        result_id: str,
        position: int,
        query: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Log de clique em resultado"""
        
        if not self.current_session_id:
            logger.warning("Clique logado sem sessão ativa")
            return
        
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            session_id=self.current_session_id,
            event_type=EventType.RESULT_CLICK,
            timestamp=datetime.now(timezone.utc),
            query=query,
            clicked_result_id=result_id,
            click_position=position,
            metadata=metadata
        )
        
        self.log_event(event)
    
    def log_feedback(
        self,
        feedback_type: FeedbackType,
        result_id: str = None,
        query: str = None,
        metadata: Dict[str, Any] = None
    ):
        """Log de feedback do usuário"""
        
        if not self.current_session_id:
            logger.warning("Feedback logado sem sessão ativa")
            return
        
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            session_id=self.current_session_id,
            event_type=EventType.RESULT_RATING,
            timestamp=datetime.now(timezone.utc),
            query=query,
            clicked_result_id=result_id,
            feedback_type=feedback_type,
            metadata=metadata
        )
        
        self.log_event(event)
    
    def log_event(self, event: TelemetryEvent):
        """Log genérico de evento"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO telemetry_events 
                    (event_id, session_id, event_type, timestamp, query, results, 
                     clicked_result_id, click_position, feedback_type, response_time_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.session_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.query,
                    json.dumps(event.results) if event.results else None,
                    event.clicked_result_id,
                    event.click_position,
                    event.feedback_type.value if event.feedback_type else None,
                    event.response_time_ms,
                    json.dumps(event.metadata) if event.metadata else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao logar evento: {e}")
    
    def _update_session_stats(self):
        """Atualiza estatísticas da sessão atual"""
        if not self.current_session_id:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            # Contar queries e cliques
            cursor = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN event_type = 'query' THEN 1 END) as total_queries,
                    COUNT(CASE WHEN event_type = 'result_click' THEN 1 END) as total_clicks,
                    AVG(CASE WHEN response_time_ms IS NOT NULL THEN response_time_ms END) as avg_response_time
                FROM telemetry_events 
                WHERE session_id = ?
            """, (self.current_session_id,))
            
            stats = cursor.fetchone()
            
            # Atualizar sessão
            conn.execute("""
                UPDATE sessions 
                SET end_time = ?, total_queries = ?, total_clicks = ?, avg_response_time = ?
                WHERE session_id = ?
            """, (
                datetime.now(timezone.utc).isoformat(),
                stats[0],
                stats[1], 
                stats[2],
                self.current_session_id
            ))
            conn.commit()


class AnalyticsEngine:
    """Motor de análise dos dados de telemetria"""
    
    def __init__(self):
        self.collector = TelemetryCollector()
        self.db_path = self.collector.db_path
    
    def get_query_analytics(self, query: str = None, days: int = 30) -> List[QueryAnalytics]:
        """
        Análise de queries
        
        Args:
            query: Query específica (None para todas)
            days: Número de dias para análise
            
        Returns:
            Lista de QueryAnalytics
        """
        
        with sqlite3.connect(self.db_path) as conn:
            # Buscar dados de queries
            base_where = "WHERE event_type = 'query' AND datetime(timestamp) > datetime('now', '-{} days')".format(days)
            if query:
                base_where += f" AND query = '{query}'"
            
            df = pd.read_sql_query(f"""
                SELECT query, timestamp, response_time_ms, event_id
                FROM telemetry_events 
                {base_where}
                ORDER BY timestamp DESC
            """, conn)
            
            if df.empty:
                return []
            
            analytics = []
            
            for query_text in df['query'].unique():
                if pd.isna(query_text):
                    continue
                
                query_df = df[df['query'] == query_text]
                
                # Estatísticas básicas
                total_searches = len(query_df)
                avg_response_time = query_df['response_time_ms'].mean()
                
                # CTR (Click Through Rate)
                query_event_ids = query_df['event_id'].tolist()
                
                click_count = pd.read_sql_query("""
                    SELECT COUNT(*) as count FROM telemetry_events 
                    WHERE event_type = 'result_click' 
                    AND session_id IN (
                        SELECT DISTINCT session_id FROM telemetry_events 
                        WHERE event_id IN ({})
                    )
                """.format(','.join([f"'{eid}'" for eid in query_event_ids])), conn).iloc[0]['count']
                
                ctr = click_count / total_searches if total_searches > 0 else 0
                
                # Rating médio
                rating_df = pd.read_sql_query("""
                    SELECT feedback_type FROM telemetry_events 
                    WHERE event_type = 'result_rating' AND query = ?
                """, conn, params=(query_text,))
                
                avg_rating = None
                if not rating_df.empty:
                    # Converter ratings para números
                    rating_map = {
                        'rating_1': 1, 'rating_2': 2, 'rating_3': 3, 
                        'rating_4': 4, 'rating_5': 5,
                        'thumbs_up': 4, 'thumbs_down': 2
                    }
                    ratings = rating_df['feedback_type'].map(rating_map).dropna()
                    if len(ratings) > 0:
                        avg_rating = ratings.mean()
                
                # Success rate (baseado em cliques + ratings positivas)
                success_rate = min(ctr + 0.3, 1.0)  # Heurística simples
                
                # Top clicked results
                top_clicked = pd.read_sql_query("""
                    SELECT clicked_result_id, COUNT(*) as click_count
                    FROM telemetry_events 
                    WHERE event_type = 'result_click' AND query = ?
                    GROUP BY clicked_result_id
                    ORDER BY click_count DESC
                    LIMIT 5
                """, conn, params=(query_text,))
                
                top_clicked_results = top_clicked.to_dict('records')
                
                analytics.append(QueryAnalytics(
                    query=query_text,
                    total_searches=total_searches,
                    avg_response_time=avg_response_time,
                    click_through_rate=ctr,
                    avg_rating=avg_rating,
                    success_rate=success_rate,
                    top_clicked_results=top_clicked_results
                ))
            
            # Ordenar por total de buscas
            analytics.sort(key=lambda x: x.total_searches, reverse=True)
            return analytics
    
    def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Métricas gerais de desempenho"""
        
        with sqlite3.connect(self.db_path) as conn:
            metrics = {}
            
            # Métricas básicas
            basic_metrics = pd.read_sql_query(f"""
                SELECT 
                    COUNT(CASE WHEN event_type = 'query' THEN 1 END) as total_queries,
                    COUNT(CASE WHEN event_type = 'result_click' THEN 1 END) as total_clicks,
                    COUNT(DISTINCT session_id) as total_sessions,
                    AVG(CASE WHEN response_time_ms IS NOT NULL THEN response_time_ms END) as avg_response_time
                FROM telemetry_events 
                WHERE datetime(timestamp) > datetime('now', '-{days} days')
            """, conn).iloc[0]
            
            metrics.update(basic_metrics.to_dict())
            
            # CTR global
            if metrics['total_queries'] > 0:
                metrics['global_ctr'] = metrics['total_clicks'] / metrics['total_queries']
            else:
                metrics['global_ctr'] = 0
            
            # Distribuição de ratings
            ratings_dist = pd.read_sql_query(f"""
                SELECT feedback_type, COUNT(*) as count
                FROM telemetry_events 
                WHERE event_type = 'result_rating' 
                AND datetime(timestamp) > datetime('now', '-{days} days')
                GROUP BY feedback_type
            """, conn)
            
            metrics['ratings_distribution'] = ratings_dist.to_dict('records')
            
            return metrics


class LearningEngine:
    """Motor de aprendizado para otimização de pesos"""
    
    def __init__(self):
        self.collector = TelemetryCollector()
        self.analytics = AnalyticsEngine()
        self.current_weights = {
            'lexical_weight': config.LEXICAL_WEIGHT,
            'vector_weight': config.VECTOR_WEIGHT,
            'bm25_weight': 0.25,
            'cosine_weight': 0.35,
            'heuristic_weight': 0.25
        }
    
    def analyze_ranking_performance(self) -> Dict[str, float]:
        """
        Analisa desempenho do ranking atual
        
        Returns:
            Métricas de desempenho
        """
        metrics = self.analytics.get_performance_metrics(days=7)
        
        # Score baseado em múltiplas métricas
        performance_score = 0.0
        
        # CTR (40% do score)
        ctr = metrics.get('global_ctr', 0)
        performance_score += ctr * 0.4
        
        # Tempo de resposta (20% do score, invertido)
        avg_time = metrics.get('avg_response_time', 1000)
        time_score = max(0, 1 - (avg_time / 2000))  # 2s = score 0
        performance_score += time_score * 0.2
        
        # Ratings positivos (40% do score)
        ratings_dist = metrics.get('ratings_distribution', [])
        positive_ratings = 0
        total_ratings = 0
        
        for rating in ratings_dist:
            count = rating['count']
            feedback_type = rating['feedback_type']
            total_ratings += count
            
            if feedback_type in ['rating_4', 'rating_5', 'thumbs_up']:
                positive_ratings += count
        
        if total_ratings > 0:
            rating_score = positive_ratings / total_ratings
            performance_score += rating_score * 0.4
        
        return {
            'performance_score': performance_score,
            'ctr': ctr,
            'avg_response_time': avg_time,
            'positive_rating_ratio': positive_ratings / total_ratings if total_ratings > 0 else 0
        }
    
    def suggest_weight_adjustments(self) -> Dict[str, float]:
        """
        Sugere ajustes nos pesos baseado no desempenho
        
        Returns:
            Novos pesos sugeridos
        """
        performance = self.analyze_ranking_performance()
        current_score = performance['performance_score']
        
        # Se o desempenho está bom (>0.7), fazer ajustes pequenos
        # Se está ruim (<0.4), fazer ajustes maiores
        
        adjustment_factor = 0.1 if current_score > 0.7 else 0.2 if current_score > 0.4 else 0.3
        
        new_weights = self.current_weights.copy()
        
        # Heurísticas de ajuste baseadas no desempenho
        if performance['ctr'] < 0.3:  # CTR baixo
            # Aumentar peso vetorial (melhor para relevância semântica)
            new_weights['vector_weight'] += adjustment_factor
            new_weights['lexical_weight'] -= adjustment_factor
        
        if performance['avg_response_time'] > 1500:  # Tempo alto
            # Aumentar peso léxico (mais rápido)
            new_weights['lexical_weight'] += adjustment_factor * 0.5
            new_weights['vector_weight'] -= adjustment_factor * 0.5
        
        # Normalizar pesos para somar 1.0
        total_weight = sum(new_weights.values())
        if total_weight != 1.0:
            for key in new_weights:
                new_weights[key] /= total_weight
        
        return new_weights
    
    def apply_weight_adjustments(self, new_weights: Dict[str, float]):
        """Aplica novos pesos e salva no banco"""
        
        # Salvar no banco para histórico
        with sqlite3.connect(self.collector.db_path) as conn:
            performance = self.analyze_ranking_performance()
            
            conn.execute("""
                INSERT INTO ranking_weights 
                (timestamp, lexical_weight, vector_weight, bm25_weight, cosine_weight, heuristic_weight, performance_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                new_weights.get('lexical_weight', 0.4),
                new_weights.get('vector_weight', 0.6), 
                new_weights.get('bm25_weight', 0.25),
                new_weights.get('cosine_weight', 0.35),
                new_weights.get('heuristic_weight', 0.25),
                performance['performance_score'],
                json.dumps(performance)
            ))
            conn.commit()
        
        # Atualizar pesos atuais
        self.current_weights.update(new_weights)
        
        logger.info(f"Pesos atualizados: {new_weights}")
    
    def auto_optimize(self, min_queries: int = 50) -> bool:
        """
        Otimização automática baseada em dados
        
        Args:
            min_queries: Mínimo de queries necessárias para otimizar
            
        Returns:
            True se otimização foi aplicada
        """
        
        # Verificar se há dados suficientes
        metrics = self.analytics.get_performance_metrics(days=7)
        
        if metrics['total_queries'] < min_queries:
            logger.info(f"Dados insuficientes para otimização: {metrics['total_queries']}/{min_queries} queries")
            return False
        
        # Analisar desempenho atual
        performance = self.analyze_ranking_performance()
        
        # Se desempenho está muito bom, não mexer
        if performance['performance_score'] > 0.8:
            logger.info("Desempenho já está ótimo, mantendo pesos atuais")
            return False
        
        # Sugerir e aplicar novos pesos
        new_weights = self.suggest_weight_adjustments()
        self.apply_weight_adjustments(new_weights)
        
        logger.info(f"Otimização automática aplicada. Novo score esperado: {performance['performance_score']:.3f}")
        return True


def main():
    """Função principal para teste"""
    
    # Criar componentes
    collector = TelemetryCollector()
    analytics = AnalyticsEngine()
    learning = LearningEngine()
    
    print("📊 Testando Telemetry & Learning")
    print("-" * 50)
    
    # Simular uma sessão de uso
    session_id = collector.start_session({"user_agent": "test", "version": "1.0"})
    
    # Simular algumas queries
    queries = [
        "Python programming",
        "Machine learning basics", 
        "Data science tutorial"
    ]
    
    for query in queries:
        # Log query
        results = [{"id": f"doc_{i}", "title": f"Result {i}"} for i in range(3)]
        query_id = collector.log_query(query, results, response_time_ms=500 + np.random.normal(0, 100))
        
        # Simular cliques
        if np.random.random() > 0.3:  # 70% chance de clique
            position = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])  # Bias para primeiros resultados
            collector.log_click(results[position]["id"], position, query)
            
            # Simular feedback
            if np.random.random() > 0.5:  # 50% chance de feedback
                feedback = np.random.choice([
                    FeedbackType.THUMBS_UP, 
                    FeedbackType.THUMBS_DOWN,
                    FeedbackType.RATING_4,
                    FeedbackType.RATING_5
                ], p=[0.4, 0.1, 0.3, 0.2])
                collector.log_feedback(feedback, results[position]["id"], query)
    
    # Finalizar sessão
    collector.end_session()
    
    print(f"✅ Sessão simulada: {session_id}")
    
    # Análises
    print("\n📈 Análises:")
    query_analytics = analytics.get_query_analytics(days=1)
    
    for qa in query_analytics:
        print(f"Query: '{qa.query}'")
        print(f"  Buscas: {qa.total_searches}")
        print(f"  CTR: {qa.click_through_rate:.2%}")
        print(f"  Tempo médio: {qa.avg_response_time:.0f}ms")
        print(f"  Rating: {qa.avg_rating:.1f}" if qa.avg_rating else "  Rating: N/A")
    
    # Métricas gerais
    print("\n📊 Métricas Gerais:")
    metrics = analytics.get_performance_metrics(days=1)
    for key, value in metrics.items():
        if key != 'ratings_distribution':
            print(f"  {key}: {value}")
    
    # Teste de aprendizado
    print("\n🧠 Learning Engine:")
    performance = learning.analyze_ranking_performance()
    print(f"Performance Score: {performance['performance_score']:.3f}")
    
    suggested_weights = learning.suggest_weight_adjustments()
    print("Pesos sugeridos:", suggested_weights)


if __name__ == "__main__":
    main()
