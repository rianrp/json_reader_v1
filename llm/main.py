"""
Módulo de LLM
Responsável por:
- Interface unificada: generate_answer(prompt)
- Suporte para múltiplos provedores (Groq, OpenAI, HuggingFace, etc.)
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

from groq import Groq
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


@dataclass
class LLMResponse:
    """Resposta do LLM"""
    content: str
    model: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]


class LLMProvider(ABC):
    """Classe abstrata para provedores de LLM"""
    
    @abstractmethod
    def generate_answer(self, prompt: str, **kwargs) -> LLMResponse:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class GroqProvider(LLMProvider):
    """Provedor Groq"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("Groq API key não fornecida")
    
    def is_available(self) -> bool:
        """Verifica se o provedor está disponível"""
        return self.client is not None and bool(self.api_key)
    
    def generate_answer(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Gera resposta usando Groq
        
        Args:
            prompt: Prompt para o modelo
            **kwargs: Parâmetros adicionais (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse com a resposta
        """
        if not self.is_available():
            raise ValueError("Groq provider não está disponível")
        
        try:
            # Parâmetros padrão
            params = {
                'model': self.model,
                'messages': [{"role": "user", "content": prompt}],
                'temperature': kwargs.get('temperature', 0.1),
                'max_tokens': kwargs.get('max_tokens', 1024),
                'top_p': kwargs.get('top_p', 1.0),
                'stream': False
            }
            
            # Fazer chamada para API
            response = self.client.chat.completions.create(**params)
            
            # Extrair resposta
            content = response.choices[0].message.content
            
            # Preparar metadata de uso
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            metadata = {
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Erro na chamada Groq: {e}")
            raise


class OpenAIProvider(LLMProvider):
    """Provedor OpenAI (placeholder)"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package não instalado")
    
    def is_available(self) -> bool:
        return self.client is not None and bool(self.api_key)
    
    def generate_answer(self, prompt: str, **kwargs) -> LLMResponse:
        """Gera resposta usando OpenAI"""
        if not self.is_available():
            raise ValueError("OpenAI provider não está disponível")
        
        # Implementação similar ao Groq
        # Por simplicidade, retorna placeholder
        return LLMResponse(
            content="[OpenAI response placeholder]",
            model=self.model,
            usage={'total_tokens': 0},
            metadata={}
        )


class HuggingFaceProvider(LLMProvider):
    """Provedor HuggingFace local (placeholder)"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.pipeline = None
        
        try:
            from transformers import pipeline
            self.pipeline = pipeline("text-generation", model=model_name)
        except ImportError:
            logger.warning("transformers package não instalado")
        except Exception as e:
            logger.warning(f"Erro ao carregar modelo HF: {e}")
    
    def is_available(self) -> bool:
        return self.pipeline is not None
    
    def generate_answer(self, prompt: str, **kwargs) -> LLMResponse:
        """Gera resposta usando HuggingFace"""
        if not self.is_available():
            raise ValueError("HuggingFace provider não está disponível")
        
        # Implementação placeholder
        return LLMResponse(
            content="[HuggingFace response placeholder]",
            model=self.model_name,
            usage={'total_tokens': 0},
            metadata={}
        )


class LLMManager:
    """Gerenciador principal de LLM com fallback"""
    
    def __init__(self):
        self.providers = {}
        self.default_provider = None
        
        self._setup_providers()
    
    def _setup_providers(self):
        """Configura os provedores disponíveis"""
        
        # Groq (principal)
        try:
            groq_provider = GroqProvider()
            if groq_provider.is_available():
                self.providers['groq'] = groq_provider
                self.default_provider = 'groq'
                logger.info("Groq provider configurado")
        except Exception as e:
            logger.warning(f"Erro ao configurar Groq: {e}")
        
        # OpenAI (fallback)
        try:
            openai_provider = OpenAIProvider()
            if openai_provider.is_available():
                self.providers['openai'] = openai_provider
                if not self.default_provider:
                    self.default_provider = 'openai'
                logger.info("OpenAI provider configurado")
        except Exception as e:
            logger.warning(f"Erro ao configurar OpenAI: {e}")
        
        # HuggingFace (fallback local)
        try:
            hf_provider = HuggingFaceProvider()
            if hf_provider.is_available():
                self.providers['huggingface'] = hf_provider
                if not self.default_provider:
                    self.default_provider = 'huggingface'
                logger.info("HuggingFace provider configurado")
        except Exception as e:
            logger.warning(f"Erro ao configurar HuggingFace: {e}")
        
        if not self.providers:
            logger.error("Nenhum provedor LLM disponível!")
    
    def generate_answer(
        self, 
        prompt: str, 
        provider: str = None,
        **kwargs
    ) -> LLMResponse:
        """
        Interface unificada para gerar respostas
        
        Args:
            prompt: Prompt para o modelo
            provider: Provedor específico ('groq', 'openai', 'huggingface')
            **kwargs: Parâmetros adicionais
            
        Returns:
            LLMResponse com a resposta gerada
        """
        
        # Determinar provedor a usar
        if provider and provider in self.providers:
            selected_provider = provider
        elif self.default_provider:
            selected_provider = self.default_provider
        else:
            raise ValueError("Nenhum provedor LLM disponível")
        
        try:
            logger.info(f"Gerando resposta com {selected_provider}")
            response = self.providers[selected_provider].generate_answer(prompt, **kwargs)
            logger.info(f"Resposta gerada: {len(response.content)} caracteres")
            return response
            
        except Exception as e:
            logger.error(f"Erro com provedor {selected_provider}: {e}")
            
            # Tentar fallback
            for fallback_provider in self.providers:
                if fallback_provider != selected_provider:
                    try:
                        logger.info(f"Tentando fallback: {fallback_provider}")
                        return self.providers[fallback_provider].generate_answer(prompt, **kwargs)
                    except Exception as fallback_e:
                        logger.warning(f"Fallback {fallback_provider} falhou: {fallback_e}")
            
            raise RuntimeError("Todos os provedores LLM falharam")
    
    def create_rag_prompt(
        self, 
        query: str, 
        context_passages: List[Dict[str, Any]],
        max_context_length: int = 2000
    ) -> str:
        """
        Cria um prompt otimizado para RAG
        
        Args:
            query: Pergunta do usuário
            context_passages: Lista de passagens relevantes
            max_context_length: Tamanho máximo do contexto
            
        Returns:
            Prompt formatado para RAG
        """
        
        # Construir contexto
        context_parts = []
        current_length = 0
        
        for i, passage in enumerate(context_passages):
            passage_text = f"[Documento {i+1}]\nTítulo: {passage.get('title', 'N/A')}\nConteúdo: {passage.get('content', '')}\n"
            
            if current_length + len(passage_text) <= max_context_length:
                context_parts.append(passage_text)
                current_length += len(passage_text)
            else:
                # Truncar último documento se necessário
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Mínimo viável
                    truncated = passage_text[:remaining_space] + "..."
                    context_parts.append(truncated)
                break
        
        context = "\n".join(context_parts)
        
        # Template do prompt
        prompt_template = """Você é um assistente AI especializado em responder perguntas com base em documentos fornecidos.

CONTEXTO:
{context}

PERGUNTA: {query}

INSTRUÇÕES:
1. Responda à pergunta usando APENAS as informações fornecidas no contexto acima
2. Se a informação não estiver disponível no contexto, diga explicitamente "Não encontrei essa informação nos documentos fornecidos"
3. Cite os documentos relevantes na sua resposta (ex: "Segundo o Documento 1...")
4. Seja preciso e objetivo
5. Mantenha a resposta focada na pergunta

RESPOSTA:"""

        return prompt_template.format(context=context, query=query)
    
    def get_available_providers(self) -> List[str]:
        """Retorna lista de provedores disponíveis"""
        return list(self.providers.keys())


def main():
    """Função principal para teste"""
    
    # Criar manager
    llm_manager = LLMManager()
    
    print("🤖 Testando LLM Manager")
    print(f"Provedores disponíveis: {llm_manager.get_available_providers()}")
    print(f"Provedor padrão: {llm_manager.default_provider}")
    print("-" * 50)
    
    # Teste de prompt simples
    test_prompt = "Explique o que é Python em 2 frases."
    
    try:
        response = llm_manager.generate_answer(test_prompt)
        print(f"📝 Resposta ({response.model}):")
        print(response.content)
        print(f"📊 Uso: {response.usage}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
    
    print()
    
    # Teste de prompt RAG
    print("🔗 Testando prompt RAG:")
    
    context_passages = [
        {
            'title': 'Python Básico',
            'content': 'Python é uma linguagem de programação interpretada e de alto nível, conhecida por sua sintaxe simples.'
        },
        {
            'title': 'Python Aplicações', 
            'content': 'Python é amplamente usado em ciência de dados, desenvolvimento web, automação e inteligência artificial.'
        }
    ]
    
    query = "Para que é usado Python?"
    
    rag_prompt = llm_manager.create_rag_prompt(query, context_passages)
    print("Prompt RAG criado:")
    print("-" * 30)
    print(rag_prompt[:500] + "..." if len(rag_prompt) > 500 else rag_prompt)


if __name__ == "__main__":
    main()
