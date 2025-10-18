"""
Sistema RAG Completo - Integração de Todos os Módulos
Arquivo principal que orquestra todo o pipeline RAG
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from loguru import logger

# Importar todos os módulos
import sys
sys.path.append(str(Path(__file__).parent))

try:
    from index.main import IndexManager
    from lexical_vector.main import HybridSearcher
    from rerank.main import Reranker
    from llm.main import LLMManager
    from grounding.main import GroundingValidator
    
    # Imports alternativos para módulos com hífens
    import importlib
    post_processing_module = importlib.import_module('post-processing.main')
    ResponseFormatter = post_processing_module.ResponseFormatter
    OutputFormat = post_processing_module.OutputFormat
    
    telemetry_module = importlib.import_module('telemetry-learning.main')
    TelemetryCollector = telemetry_module.TelemetryCollector
    FeedbackType = telemetry_module.FeedbackType  
    LearningEngine = telemetry_module.LearningEngine
    
    from config import config, setup_directories
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Verifique se todas as dependências estão instaladas")
    sys.exit(1)


@dataclass 
class RAGResponse:
    """Resposta completa do sistema RAG"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    response_time_ms: float
    is_grounded: bool
    query_id: str
    metadata: Dict[str, Any]


class RAGSystem:
    """Sistema RAG completo"""
    
    def __init__(self, enable_telemetry: bool = True):
        """
        Inicializa o sistema RAG
        
        Args:
            enable_telemetry: Se deve habilitar coleta de telemetria
        """
        setup_directories()
        
        self.enable_telemetry = enable_telemetry
        
        # Inicializar componentes
        logger.info("Inicializando sistema RAG...")
        
        try:
            self.index_manager = IndexManager()
            self.searcher = HybridSearcher()
            self.reranker = Reranker()
            self.llm_manager = LLMManager()
            self.grounding_validator = GroundingValidator()
            self.response_formatter = ResponseFormatter()
            
            if self.enable_telemetry:
                self.telemetry_collector = TelemetryCollector()
                self.learning_engine = LearningEngine()
            else:
                self.telemetry_collector = None
                self.learning_engine = None
            
            logger.info("✅ Sistema RAG inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar sistema RAG: {e}")
            raise
    
    def index_documents(self, json_path: str) -> bool:
        """
        Indexa documentos de um arquivo JSON
        
        Args:
            json_path: Caminho para arquivo JSON com documentos
            
        Returns:
            True se bem-sucedido
        """
        logger.info(f"Indexando documentos de {json_path}")
        start_time = time.time()
        
        try:
            success = self.index_manager.build_indices_from_json(json_path)
            
            if success:
                # Treinar BM25 do reranker com documentos indexados
                documents = self.index_manager.ingestion.get_all_documents()
                self.reranker.fit_bm25(documents)
                
                elapsed_time = (time.time() - start_time) * 1000
                logger.info(f"✅ Indexação completa em {elapsed_time:.0f}ms")
                
                return True
            else:
                logger.error("❌ Falha na indexação")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro durante indexação: {e}")
            return False
    
    def query(
        self, 
        question: str,
        top_k: int = None,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        include_sources: bool = True,
        provider: str = None
    ) -> RAGResponse:
        """
        Executa query completa no sistema RAG
        
        Args:
            question: Pergunta do usuário
            top_k: Número máximo de resultados
            output_format: Formato de saída
            include_sources: Se deve incluir fontes
            provider: Provedor LLM específico
            
        Returns:
            RAGResponse completa
        """
        start_time = time.time()
        query_id = None
        
        if top_k is None:
            top_k = config.TOP_K_FINAL
        
        try:
            logger.info(f"🔍 Processando query: '{question}'")
            
            # 1. Telemetria: Iniciar sessão se necessário
            if self.telemetry_collector and not self.telemetry_collector.current_session_id:
                self.telemetry_collector.start_session()
            
            # 2. Busca híbrida (léxica + vetorial)
            logger.info("📚 Executando busca híbrida...")
            search_results = self.searcher.search(question, top_k=top_k * 2)  # Buscar mais para rerank
            
            if not search_results:
                logger.warning("⚠️ Nenhum resultado encontrado na busca")
                return self._create_empty_response(question, start_time)
            
            logger.info(f"📊 Encontrados {len(search_results)} resultados iniciais")
            
            # 3. Reranking
            logger.info("🔄 Aplicando reranking...")
            reranked_results = self.reranker.rerank(question, search_results, top_k=top_k)
            
            if not reranked_results:
                logger.warning("⚠️ Nenhum resultado após reranking")
                return self._create_empty_response(question, start_time)
            
            logger.info(f"📈 {len(reranked_results)} resultados após reranking")
            
            # 4. Preparar contexto para LLM
            context_passages = []
            for result in reranked_results:
                context_passages.append({
                    'title': result.title,
                    'content': result.content,
                    'id': result.doc_id
                })
            
            # 5. Gerar resposta com LLM
            logger.info("🤖 Gerando resposta com LLM...")
            rag_prompt = self.llm_manager.create_rag_prompt(question, context_passages)
            
            llm_start = time.time()
            llm_response = self.llm_manager.generate_answer(rag_prompt, provider=provider)
            llm_time = (time.time() - llm_start) * 1000
            
            logger.info(f"💬 Resposta gerada em {llm_time:.0f}ms")
            
            # 6. Validação de grounding
            logger.info("🔍 Validando grounding...")
            grounding_result = self.grounding_validator.validate_grounding(
                llm_response.content, context_passages
            )
            
            # 7. Pós-processamento
            logger.info("✨ Aplicando pós-processamento...")
            processed_response = self.response_formatter.format_response(
                llm_response.content,
                grounding_result=grounding_result,
                format_type=output_format,
                include_sources=include_sources
            )
            
            # 8. Telemetria: Log da query
            if self.telemetry_collector:
                response_time = (time.time() - start_time) * 1000
                query_id = self.telemetry_collector.log_query(
                    question, 
                    [{'id': r.doc_id, 'title': r.title, 'score': r.final_score} for r in reranked_results],
                    response_time,
                    {
                        'llm_provider': llm_response.model,
                        'grounded': grounding_result.is_grounded,
                        'confidence': grounding_result.confidence_score,
                        'top_k': top_k
                    }
                )
            
            # 9. Construir resposta final
            total_time = (time.time() - start_time) * 1000
            
            rag_response = RAGResponse(
                answer=processed_response.content,
                sources=processed_response.sources,
                confidence_score=grounding_result.confidence_score,
                response_time_ms=total_time,
                is_grounded=grounding_result.is_grounded,
                query_id=query_id or "unknown",
                metadata={
                    'search_results_count': len(search_results),
                    'reranked_count': len(reranked_results),
                    'llm_provider': llm_response.model,
                    'llm_usage': llm_response.usage,
                    'grounding_metadata': grounding_result.metadata,
                    'quality_score': processed_response.quality_score,
                    'output_format': output_format.value
                }
            )
            
            logger.info(f"✅ Query processada em {total_time:.0f}ms - Grounded: {grounding_result.is_grounded}")
            return rag_response
            
        except Exception as e:
            logger.error(f"❌ Erro durante processamento da query: {e}")
            return self._create_error_response(question, str(e), start_time)
    
    def _create_empty_response(self, question: str, start_time: float) -> RAGResponse:
        """Cria resposta vazia quando não há resultados"""
        return RAGResponse(
            answer="Desculpe, não consegui encontrar informações relevantes para responder sua pergunta nos documentos disponíveis.",
            sources=[],
            confidence_score=0.0,
            response_time_ms=(time.time() - start_time) * 1000,
            is_grounded=False,
            query_id="empty",
            metadata={'error': 'no_results_found'}
        )
    
    def _create_error_response(self, question: str, error: str, start_time: float) -> RAGResponse:
        """Cria resposta de erro"""
        return RAGResponse(
            answer=f"Ocorreu um erro ao processar sua pergunta: {error}",
            sources=[],
            confidence_score=0.0,
            response_time_ms=(time.time() - start_time) * 1000,
            is_grounded=False,
            query_id="error",
            metadata={'error': error}
        )
    
    def log_user_feedback(
        self, 
        query_id: str, 
        feedback_type: FeedbackType,
        result_id: str = None
    ):
        """
        Log de feedback do usuário
        
        Args:
            query_id: ID da query
            feedback_type: Tipo de feedback
            result_id: ID do resultado (opcional)
        """
        if self.telemetry_collector:
            self.telemetry_collector.log_feedback(feedback_type, result_id, query_id)
    
    def log_result_click(
        self,
        result_id: str,
        position: int,
        query: str = None
    ):
        """
        Log de clique em resultado
        
        Args:
            result_id: ID do resultado clicado
            position: Posição do resultado (0-indexed)
            query: Query original (opcional)
        """
        if self.telemetry_collector:
            self.telemetry_collector.log_click(result_id, position, query)
    
    def optimize_performance(self) -> bool:
        """
        Executa otimização automática baseada em telemetria
        
        Returns:
            True se otimização foi aplicada
        """
        if not self.learning_engine:
            logger.warning("Learning engine não está habilitado")
            return False
        
        return self.learning_engine.auto_optimize()
    
    def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """
        Obtém analytics do sistema
        
        Args:
            days: Número de dias para análise
            
        Returns:
            Dict com métricas e análises
        """
        if not self.telemetry_collector:
            return {"error": "Telemetria não habilitada"}
        
        import importlib
        telemetry_module = importlib.import_module('telemetry-learning.main')
        AnalyticsEngine = telemetry_module.AnalyticsEngine
        analytics = AnalyticsEngine()
        
        return {
            'performance_metrics': analytics.get_performance_metrics(days),
            'query_analytics': analytics.get_query_analytics(days=days),
            'system_status': self._get_system_status()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Obtém status dos componentes do sistema"""
        status = {
            'index_available': True,  # TODO: implementar verificação real
            'llm_providers': self.llm_manager.get_available_providers(),
            'telemetry_enabled': self.enable_telemetry,
            'current_session': self.telemetry_collector.current_session_id if self.telemetry_collector else None
        }
        return status


def main():
    """Função principal para demonstração"""
    
    print("🚀 Iniciando Sistema RAG Completo")
    print("=" * 60)
    
    # Inicializar sistema
    rag = RAGSystem(enable_telemetry=True)
    
    # Verificar se há documentos indexados, senão criar alguns de exemplo
    sample_json = Path("sample_documents.json")
    
    if not sample_json.exists():
        print("📄 Criando documentos de exemplo...")
        sample_docs = [
            {
                "title": "Introdução ao Python",
                "content": "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral. Foi criada por Guido van Rossum e lançada em 1991. Python é conhecida por sua sintaxe clara e legível, o que a torna uma excelente escolha para iniciantes em programação. A linguagem suporta múltiplos paradigmas de programação, incluindo programação orientada a objetos, programação procedural e programação funcional.",
                "category": "programming",
                "difficulty": "beginner"
            },
            {
                "title": "Machine Learning com Python",
                "content": "Machine Learning é um subcampo da inteligência artificial que permite que computadores aprendam e melhorem automaticamente a partir da experiência sem serem explicitamente programados. Python oferece várias bibliotecas poderosas para ML, incluindo scikit-learn, TensorFlow e PyTorch. Estas ferramentas facilitam a implementação de algoritmos de aprendizado supervisionado, não supervisionado e por reforço.",
                "category": "ai",
                "difficulty": "intermediate"
            },
            {
                "title": "Desenvolvimento Web com Python",
                "content": "Python é amplamente usado no desenvolvimento web através de frameworks como Django e Flask. Django é um framework web de alto nível que encoraja o desenvolvimento rápido e design limpo e pragmático. Flask é um micro-framework que fornece as ferramentas básicas para construir aplicações web. Ambos são excelentes opções para criar desde simples sites até aplicações web complexas.",
                "category": "web",
                "difficulty": "intermediate"
            },
            {
                "title": "Ciência de Dados com Python",
                "content": "Python se tornou a linguagem preferida para ciência de dados devido à sua simplicidade e ao ecossistema rico de bibliotecas especializadas. Pandas é usado para manipulação e análise de dados, NumPy para computação numérica, Matplotlib e Seaborn para visualização de dados, e Jupyter Notebooks fornece um ambiente interativo para análise exploratória de dados.",
                "category": "data_science", 
                "difficulty": "intermediate"
            }
        ]
        
        with open(sample_json, 'w', encoding='utf-8') as f:
            json.dump(sample_docs, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Arquivo criado: {sample_json}")
    
    # Indexar documentos
    print("📚 Indexando documentos...")
    success = rag.index_documents(str(sample_json))
    
    if not success:
        print("❌ Falha na indexação. Verifique os logs.")
        return
    
    print("✅ Documentos indexados com sucesso!")
    print()
    
    # Exemplos de queries
    queries = [
        "O que é Python?",
        "Como fazer machine learning com Python?",
        "Quais frameworks web Python?",
        "Python para ciência de dados"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"🔍 Query {i}: {query}")
        print("-" * 40)
        
        # Executar query
        response = rag.query(
            query, 
            top_k=3,
            output_format=OutputFormat.MARKDOWN,
            include_sources=True
        )
        
        # Mostrar resultado
        print(f"⏱️  Tempo: {response.response_time_ms:.0f}ms")
        print(f"🎯 Confiança: {response.confidence_score:.3f}")
        print(f"✅ Grounded: {response.is_grounded}")
        print()
        
        print("📝 Resposta:")
        print(response.answer)
        print()
        
        # Simular feedback (apenas para demonstração)
        if response.confidence_score > 0.5:
            rag.log_user_feedback(response.query_id, FeedbackType.THUMBS_UP)
        
        print("=" * 60)
        print()
    
    # Mostrar analytics
    print("📊 Analytics do Sistema:")
    analytics = rag.get_analytics(days=1)
    
    if 'performance_metrics' in analytics:
        metrics = analytics['performance_metrics']
        print(f"Total de queries: {metrics.get('total_queries', 0)}")
        print(f"Total de cliques: {metrics.get('total_clicks', 0)}")
        print(f"CTR global: {metrics.get('global_ctr', 0):.2%}")
        print(f"Tempo médio: {metrics.get('avg_response_time', 0):.0f}ms")
    
    print()
    
    # Testar otimização
    print("🧠 Testando otimização automática:")
    optimized = rag.optimize_performance()
    print(f"Otimização aplicada: {'Sim' if optimized else 'Não'}")
    
    print()
    print("🎉 Demonstração completa!")


if __name__ == "__main__":
    main()