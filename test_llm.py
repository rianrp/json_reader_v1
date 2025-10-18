"""
Teste rápido do LLM com modelo atualizado
"""

from llm.main import LLMManager

def test_groq():
    """Testa o provider Groq com modelo atualizado"""
    
    llm = LLMManager()
    
    print("Provedores disponíveis:", llm.get_available_providers())
    
    try:
        response = llm.generate_answer("Explique Python em uma frase.", provider="groq")
        print("✅ Groq funcionando!")
        print("Resposta:", response.content)
        print("Modelo:", response.model)
        print("Tokens:", response.usage)
    except Exception as e:
        print("❌ Erro no Groq:", e)
        
        # Testar fallback
        try:
            response = llm.generate_answer("Explique Python em uma frase.")
            print("✅ Fallback funcionando!")
            print("Resposta:", response.content)
        except Exception as e2:
            print("❌ Erro no fallback:", e2)

if __name__ == "__main__":
    test_groq()