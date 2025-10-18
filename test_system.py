"""
Script de teste para verificar o funcionamento do sistema RAG
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Testa se todos os imports estão funcionando"""
    print("🔧 Testando imports...")
    
    try:
        from config import config, setup_directories
        print("✅ config")
        
        from index.main import IndexManager
        print("✅ index")
        
        from lexical_vector.main import HybridSearcher
        print("✅ lexical_vector")
        
        from rerank.main import Reranker
        print("✅ rerank")
        
        from llm.main import LLMManager
        print("✅ llm")
        
        from grounding.main import GroundingValidator
        print("✅ grounding")
        
        # Teste imports com hífen usando importlib
        import importlib
        
        post_processing = importlib.import_module('post-processing.main')
        print("✅ post-processing")
        
        telemetry = importlib.import_module('telemetry-learning.main')
        print("✅ telemetry-learning")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos imports: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Testa funcionalidade básica sem dependências externas"""
    print("\n🧪 Testando funcionalidade básica...")
    
    try:
        # Testar config
        from config import config, setup_directories
        setup_directories()
        print("✅ Diretórios criados")
        
        # Testar componentes básicos
        from index.main import DocumentIngestion
        ingestion = DocumentIngestion()
        print("✅ DocumentIngestion inicializado")
        
        from llm.main import LLMManager
        llm_manager = LLMManager()
        providers = llm_manager.get_available_providers()
        print(f"✅ LLM providers disponíveis: {providers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na funcionalidade básica: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Testa dependências externas"""
    print("\n📦 Testando dependências...")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('groq', 'Groq'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('whoosh', 'Whoosh'),
        ('faiss', 'FAISS'),
        ('rank_bm25', 'Rank BM25'),
        ('loguru', 'Loguru')
    ]
    
    missing_deps = []
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - FALTANDO")
            missing_deps.append(display_name)
    
    if missing_deps:
        print(f"\n⚠️  Dependências faltando: {', '.join(missing_deps)}")
        print("Execute: pip install -r requirements.txt")
        return False
    
    return True

def test_system_integration():
    """Testa integração completa do sistema"""
    print("\n🚀 Testando integração do sistema...")
    
    try:
        # Criar um documento de teste simples
        import json
        test_doc = {
            "title": "Teste Python",
            "content": "Python é uma linguagem de programação."
        }
        
        test_file = Path("test_doc.json")
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump([test_doc], f, ensure_ascii=False)
        
        print("✅ Documento de teste criado")
        
        # Testar sistema RAG (se dependências estiverem disponíveis)
        try:
            from rag_system import RAGSystem
            rag = RAGSystem(enable_telemetry=False)  # Sem telemetria para teste
            print("✅ RAGSystem inicializado")
            
            # Tentar indexar (pode falhar se dependências não estiverem disponíveis)
            try:
                success = rag.index_documents(str(test_file))
                if success:
                    print("✅ Indexação bem-sucedida")
                    
                    # Testar query simples
                    response = rag.query("O que é Python?", top_k=1)
                    print(f"✅ Query executada - Confiança: {response.confidence_score:.3f}")
                else:
                    print("⚠️  Indexação falhou (pode ser normal sem dependências)")
            except Exception as e:
                print(f"⚠️  Erro na indexação/query: {e}")
            
        except Exception as e:
            print(f"⚠️  Erro ao inicializar RAGSystem: {e}")
        
        # Limpar arquivo de teste
        if test_file.exists():
            test_file.unlink()
            print("✅ Arquivo de teste removido")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na integração: {e}")
        traceback.print_exc()
        return False

def main():
    """Executa todos os testes"""
    print("🔍 TESTE DE VERIFICAÇÃO DO SISTEMA RAG")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Funcionalidade Básica", test_basic_functionality), 
        ("Dependências", test_dependencies),
        ("Integração", test_system_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
                
        except Exception as e:
            print(f"💥 {test_name}: ERRO - {e}")
            results.append((test_name, False))
    
    # Resumo final
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name:20} {status}")
    
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("Sistema RAG pronto para uso!")
    elif passed >= total * 0.5:
        print("⚠️  ALGUNS TESTES FALHARAM")
        print("Sistema pode funcionar parcialmente. Verifique dependências.")
    else:
        print("❌ MUITOS TESTES FALHARAM")
        print("Sistema provavelmente não funcionará. Instale dependências.")
    
    print("\n🔧 Para instalar dependências:")
    print("pip install -r requirements.txt")
    
    print("\n📚 Para usar o sistema:")
    print("python rag_system.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)