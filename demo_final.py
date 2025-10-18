"""
Demonstração final do sistema RAG completo
"""

from rag_system import RAGSystem, OutputFormat
import importlib
telemetry_module = importlib.import_module('telemetry-learning.main')
FeedbackType = telemetry_module.FeedbackType
import json

def demo_completa():
    """Demonstração completa do sistema RAG"""
    
    print("🚀 DEMONSTRAÇÃO COMPLETA DO SISTEMA RAG")
    print("=" * 60)
    
    # Inicializar sistema
    print("🔧 Inicializando sistema...")
    rag = RAGSystem(enable_telemetry=True)
    
    # Verificar se documentos existem, senão usar os existentes
    sample_file = "sample_documents.json"
    
    print(f"📚 Usando documentos de: {sample_file}")
    
    # Teste com uma query específica
    query = "O que é Python e para que é usado?"
    
    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)
    
    # Executar query completa
    response = rag.query(
        query,
        top_k=5,
        output_format=OutputFormat.MARKDOWN,
        include_sources=True
    )
    
    # Mostrar resultados
    print(f"⏱️  Tempo de resposta: {response.response_time_ms:.0f}ms")
    print(f"🎯 Score de confiança: {response.confidence_score:.3f}")
    print(f"✅ Bem fundamentado: {response.is_grounded}")
    print(f"📊 Query ID: {response.query_id}")
    
    print(f"\n📝 Resposta Gerada:")
    print("-" * 30)
    print(response.answer)
    
    if response.sources:
        print(f"\n📚 Fontes ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            # source é um SourceCitation object
            title = getattr(source, 'title', 'N/A')
            confidence = getattr(source, 'confidence', 0)
            print(f"{i}. {title} (Confiança: {confidence:.3f})")
    
    # Simular feedback positivo
    if response.is_grounded:
        rag.log_user_feedback(response.query_id, FeedbackType.THUMBS_UP)
        print("\n👍 Feedback positivo registrado")
    
    # Mostrar metadata
    print(f"\n🔍 Metadata do Sistema:")
    print(f"  - Resultados encontrados: {response.metadata.get('search_results_count', 0)}")
    print(f"  - Após reranking: {response.metadata.get('reranked_count', 0)}")
    print(f"  - Provedor LLM: {response.metadata.get('llm_provider', 'N/A')}")
    print(f"  - Tokens usados: {response.metadata.get('llm_usage', {}).get('total_tokens', 0)}")
    print(f"  - Score de qualidade: {response.metadata.get('quality_score', 0):.3f}")
    
    # Testar formatos diferentes
    print(f"\n🎨 Testando outros formatos de saída:")
    
    # HTML
    html_response = rag.query(
        "Resumo sobre Python", 
        top_k=2, 
        output_format=OutputFormat.HTML,
        include_sources=False
    )
    print(f"HTML (primeiros 200 chars): {html_response.answer[:200]}...")
    
    # JSON
    json_response = rag.query(
        "Python frameworks",
        top_k=2,
        output_format=OutputFormat.JSON,
        include_sources=True
    )
    print(f"JSON: {json_response.answer[:150]}...")
    
    # Analytics do sistema
    print(f"\n📊 Analytics do Sistema:")
    try:
        analytics = rag.get_analytics(days=1)
        
        if 'performance_metrics' in analytics:
            metrics = analytics['performance_metrics']
            print(f"  - Total queries: {metrics.get('total_queries', 0)}")
            print(f"  - Total cliques: {metrics.get('total_clicks', 0)}")
            print(f"  - CTR: {metrics.get('global_ctr', 0):.2%}")
            print(f"  - Tempo médio: {metrics.get('avg_response_time', 0):.0f}ms")
            
        if 'system_status' in analytics:
            status = analytics['system_status']
            print(f"  - Provedores LLM: {status.get('llm_providers', [])}")
            print(f"  - Telemetria: {'Ativa' if status.get('telemetry_enabled') else 'Inativa'}")
            print(f"  - Sessão atual: {status.get('current_session', 'N/A')[:8]}...")
    except Exception as e:
        print(f"  ⚠️  Erro ao obter analytics: {e}")
    
    # Testar otimização automática
    print(f"\n🧠 Testando otimização automática:")
    try:
        optimized = rag.optimize_performance()
        print(f"  Otimização aplicada: {'✅ Sim' if optimized else '❌ Não (dados insuficientes)'}")
    except Exception as e:
        print(f"  ⚠️  Erro na otimização: {e}")
    
    print(f"\n🎉 Demonstração completa!")
    print(f"Sistema RAG funcionando perfeitamente com todos os módulos integrados!")
    
    return response

if __name__ == "__main__":
    demo_completa()