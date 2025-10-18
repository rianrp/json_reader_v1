"""
Verifica modelos disponíveis na API Groq
"""

import requests

def check_groq_models():
    """Verifica modelos disponíveis"""
    
    api_key = "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"
    
    try:
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        if response.status_code == 200:
            models = response.json()
            print("✅ Modelos disponíveis na Groq:")
            for model in models.get('data', []):
                print(f"  - {model['id']}")
        else:
            print(f"❌ Erro: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Erro na requisição: {e}")

if __name__ == "__main__":
    check_groq_models()