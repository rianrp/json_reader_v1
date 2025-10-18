"""
Módulo de Pós-processamento
Responsável por:
- Formatação de respostas
- Destacar fontes
- Remover conteúdo indesejado (lixo)
"""

import re
import html
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


class OutputFormat(Enum):
    """Formatos de saída disponíveis"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


@dataclass
class SourceCitation:
    """Citação de fonte"""
    doc_id: str
    title: str
    snippet: str
    confidence: float
    position: int  # Posição na resposta onde deve aparecer


@dataclass
class ProcessedResponse:
    """Resposta processada"""
    content: str
    format: OutputFormat
    sources: List[SourceCitation]
    metadata: Dict[str, Any]
    quality_score: float


class TextCleaner:
    """Responsável por limpeza de texto"""
    
    def __init__(self):
        # Padrões para remover
        self.unwanted_patterns = [
            r'\[?\s*AI\s*\]?',  # Marcadores AI
            r'\[?\s*Assistant\s*\]?',  # Marcadores Assistant
            r'\[?\s*GPT\s*\]?',  # Marcadores GPT
            r'Como um AI|Como uma IA',  # Frases de IA
            r'Eu sou um|Eu sou uma',  # Auto-referências
            r'\b(clique aqui|click here)\b',  # Comandos de UI
            r'```\w*\n.*?\n```',  # Blocos de código indesejados
            r'#{1,6}\s*$',  # Headers vazios markdown
        ]
        
        # Padrões de repetição
        self.repetition_patterns = [
            r'(.{10,}?)\1{2,}',  # Repetições de texto
            r'(\b\w+\b)(\s+\1){3,}',  # Palavras repetidas
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Limpa texto removendo conteúdo indesejado
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo
        """
        cleaned = text
        
        # Remover padrões indesejados
        for pattern in self.unwanted_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remover repetições
        for pattern in self.repetition_patterns:
            cleaned = re.sub(pattern, r'\1', cleaned)
        
        # Limpezas gerais
        cleaned = self._general_cleanup(cleaned)
        
        return cleaned.strip()
    
    def _general_cleanup(self, text: str) -> str:
        """Limpezas gerais de texto"""
        
        # Remover múltiplas quebras de linha
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remover espaços duplos
        text = re.sub(r' {2,}', ' ', text)
        
        # Remover espaços no início/fim de linhas
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remover linhas vazias no início/fim
        text = text.strip()
        
        return text
    
    def remove_hallucinations(self, text: str) -> str:
        """
        Remove indicações comuns de alucinação
        
        Args:
            text: Texto a ser processado
            
        Returns:
            Texto sem alucinações óbvias
        """
        # Padrões de alucinação
        hallucination_patterns = [
            r'Baseado no meu conhecimento.*?',
            r'Segundo minha base de dados.*?',
            r'De acordo com meu treinamento.*?',
            r'Não posso fornecer informações.*?sobre.*?pois.*?',
            r'Desculpe, mas não posso.*?',
        ]
        
        cleaned = text
        for pattern in hallucination_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        return cleaned.strip()


class SourceFormatter:
    """Responsável por formatação de citações"""
    
    def __init__(self):
        pass
    
    def extract_citations_from_grounding(
        self, 
        grounding_result: Any,  # GroundingResult
        max_citations: int = 5
    ) -> List[SourceCitation]:
        """
        Extrai citações do resultado de grounding
        
        Args:
            grounding_result: Resultado do grounding
            max_citations: Máximo de citações
            
        Returns:
            Lista de SourceCitation
        """
        citations = []
        
        # Extrair das passagens de suporte
        if hasattr(grounding_result, 'supporting_passages'):
            for i, passage in enumerate(grounding_result.supporting_passages[:max_citations]):
                citation = SourceCitation(
                    doc_id=passage.source_doc_id,
                    title=passage.source_title,
                    snippet=self._create_snippet(passage.text),
                    confidence=grounding_result.cosine_similarities[i] if i < len(grounding_result.cosine_similarities) else 0.0,
                    position=0  # Será calculado depois
                )
                citations.append(citation)
        
        # Extrair de string matches (backup)
        if not citations and hasattr(grounding_result, 'string_matches'):
            for i, match in enumerate(grounding_result.string_matches[:max_citations]):
                if 'passage' in match:
                    passage = match['passage']
                    citation = SourceCitation(
                        doc_id=passage.source_doc_id,
                        title=passage.source_title,
                        snippet=self._create_snippet(passage.text),
                        confidence=match['best_score'],
                        position=0
                    )
                    citations.append(citation)
        
        return citations
    
    def _create_snippet(self, text: str, max_length: int = 150) -> str:
        """Cria snippet de texto"""
        if len(text) <= max_length:
            return text
        
        # Tenta cortar em uma frase completa
        truncated = text[:max_length]
        last_sentence_end = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )
        
        if last_sentence_end > max_length * 0.7:  # Se tem pelo menos 70% do tamanho
            return text[:last_sentence_end + 1]
        
        # Senão, corta em espaço
        last_space = truncated.rfind(' ')
        if last_space > 0:
            return text[:last_space] + "..."
        
        return text[:max_length] + "..."
    
    def format_sources_markdown(self, citations: List[SourceCitation]) -> str:
        """Formata fontes em Markdown"""
        if not citations:
            return ""
        
        lines = ["\n## Fontes"]
        
        for i, citation in enumerate(citations, 1):
            confidence_emoji = "🎯" if citation.confidence > 0.8 else "📖" if citation.confidence > 0.6 else "📝"
            lines.append(f"{i}. {confidence_emoji} **{citation.title}** (Confiança: {citation.confidence:.2f})")
            lines.append(f"   > {citation.snippet}")
            lines.append("")  # Linha vazia
        
        return "\n".join(lines)
    
    def format_sources_html(self, citations: List[SourceCitation]) -> str:
        """Formata fontes em HTML"""
        if not citations:
            return ""
        
        html_parts = ["<div class='sources'>", "<h3>Fontes</h3>", "<ol>"]
        
        for citation in citations:
            confidence_class = "high" if citation.confidence > 0.8 else "medium" if citation.confidence > 0.6 else "low"
            html_parts.append(f"<li class='source {confidence_class}'>")
            html_parts.append(f"<strong>{html.escape(citation.title)}</strong>")
            html_parts.append(f"<span class='confidence'>(Confiança: {citation.confidence:.2f})</span>")
            html_parts.append(f"<blockquote>{html.escape(citation.snippet)}</blockquote>")
            html_parts.append("</li>")
        
        html_parts.extend(["</ol>", "</div>"])
        return "\n".join(html_parts)
    
    def format_sources_text(self, citations: List[SourceCitation]) -> str:
        """Formata fontes em texto simples"""
        if not citations:
            return ""
        
        lines = ["\nFONTES:"]
        
        for i, citation in enumerate(citations, 1):
            lines.append(f"{i}. {citation.title} (Confiança: {citation.confidence:.2f})")
            lines.append(f"   \"{citation.snippet}\"")
            lines.append("")
        
        return "\n".join(lines)


class ResponseFormatter:
    """Responsável por formatação da resposta completa"""
    
    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.source_formatter = SourceFormatter()
    
    def format_response(
        self, 
        raw_answer: str,
        grounding_result: Any = None,
        format_type: OutputFormat = OutputFormat.MARKDOWN,
        include_sources: bool = True,
        include_metadata: bool = False
    ) -> ProcessedResponse:
        """
        Formata resposta completa
        
        Args:
            raw_answer: Resposta bruta do LLM
            grounding_result: Resultado do grounding
            format_type: Formato de saída
            include_sources: Se deve incluir fontes
            include_metadata: Se deve incluir metadata
            
        Returns:
            ProcessedResponse formatada
        """
        
        # 1. Limpeza do texto
        cleaned_answer = self.text_cleaner.clean_text(raw_answer)
        cleaned_answer = self.text_cleaner.remove_hallucinations(cleaned_answer)
        
        # 2. Extrair citações
        citations = []
        if grounding_result and include_sources:
            citations = self.source_formatter.extract_citations_from_grounding(grounding_result)
        
        # 3. Formatação específica por tipo
        if format_type == OutputFormat.MARKDOWN:
            formatted_content = self._format_markdown(cleaned_answer, citations if include_sources else [])
        elif format_type == OutputFormat.HTML:
            formatted_content = self._format_html(cleaned_answer, citations if include_sources else [])
        elif format_type == OutputFormat.TEXT:
            formatted_content = self._format_text(cleaned_answer, citations if include_sources else [])
        elif format_type == OutputFormat.JSON:
            formatted_content = self._format_json(cleaned_answer, citations if include_sources else [])
        else:
            formatted_content = cleaned_answer
        
        # 4. Calcular qualidade
        quality_score = self._calculate_quality_score(
            cleaned_answer, raw_answer, grounding_result
        )
        
        # 5. Metadata
        metadata = {}
        if include_metadata:
            metadata = self._create_metadata(
                raw_answer, cleaned_answer, grounding_result, citations
            )
        
        return ProcessedResponse(
            content=formatted_content,
            format=format_type,
            sources=citations,
            metadata=metadata,
            quality_score=quality_score
        )
    
    def _format_markdown(self, answer: str, citations: List[SourceCitation]) -> str:
        """Formatar em Markdown"""
        formatted = answer
        
        # Adicionar fontes
        if citations:
            sources_md = self.source_formatter.format_sources_markdown(citations)
            formatted += sources_md
        
        return formatted
    
    def _format_html(self, answer: str, citations: List[SourceCitation]) -> str:
        """Formatar em HTML"""
        # Escapar HTML
        formatted = html.escape(answer)
        
        # Converter quebras de linha
        formatted = formatted.replace('\n', '<br>\n')
        
        # Adicionar fontes
        if citations:
            sources_html = self.source_formatter.format_sources_html(citations)
            formatted += sources_html
        
        return f"<div class='answer'>{formatted}</div>"
    
    def _format_text(self, answer: str, citations: List[SourceCitation]) -> str:
        """Formatar em texto simples"""
        formatted = answer
        
        # Adicionar fontes
        if citations:
            sources_text = self.source_formatter.format_sources_text(citations)
            formatted += sources_text
        
        return formatted
    
    def _format_json(self, answer: str, citations: List[SourceCitation]) -> str:
        """Formatar em JSON"""
        import json
        
        response_dict = {
            "answer": answer,
            "sources": [
                {
                    "doc_id": citation.doc_id,
                    "title": citation.title,
                    "snippet": citation.snippet,
                    "confidence": citation.confidence
                }
                for citation in citations
            ]
        }
        
        return json.dumps(response_dict, ensure_ascii=False, indent=2)
    
    def _calculate_quality_score(
        self, 
        cleaned_answer: str, 
        raw_answer: str,
        grounding_result: Any
    ) -> float:
        """Calcula score de qualidade da resposta"""
        
        score = 0.5  # Base score
        
        # 1. Penalizar se muito conteúdo foi removido na limpeza
        if len(raw_answer) > 0:
            retention_ratio = len(cleaned_answer) / len(raw_answer)
            if retention_ratio < 0.7:  # Muito conteúdo removido
                score -= 0.2
            elif retention_ratio > 0.95:  # Pouca limpeza necessária
                score += 0.1
        
        # 2. Bonus para respostas bem estruturadas
        if len(cleaned_answer.split('.')) >= 2:  # Múltiplas sentenças
            score += 0.1
        
        # 3. Bonus baseado no grounding
        if grounding_result:
            if hasattr(grounding_result, 'is_grounded') and grounding_result.is_grounded:
                score += 0.3
            if hasattr(grounding_result, 'confidence_score'):
                score += grounding_result.confidence_score * 0.2
        
        # 4. Penalizar respostas muito curtas ou muito longas
        answer_length = len(cleaned_answer)
        if 50 <= answer_length <= 1000:  # Tamanho ideal
            score += 0.1
        elif answer_length < 20:  # Muito curta
            score -= 0.3
        elif answer_length > 2000:  # Muito longa
            score -= 0.1
        
        return min(max(score, 0.0), 1.0)  # Clamp entre 0 e 1
    
    def _create_metadata(
        self, 
        raw_answer: str, 
        cleaned_answer: str,
        grounding_result: Any,
        citations: List[SourceCitation]
    ) -> Dict[str, Any]:
        """Cria metadata da resposta processada"""
        
        metadata = {
            'original_length': len(raw_answer),
            'cleaned_length': len(cleaned_answer),
            'reduction_ratio': 1 - (len(cleaned_answer) / len(raw_answer)) if raw_answer else 0,
            'citations_count': len(citations),
            'processing_timestamp': None,  # Seria preenchido com datetime
        }
        
        if grounding_result:
            if hasattr(grounding_result, 'metadata'):
                metadata.update({f'grounding_{k}': v for k, v in grounding_result.metadata.items()})
        
        return metadata


def main():
    """Função principal para teste"""
    
    # Exemplo de resposta bruta (com "lixo")
    raw_answer = """
    Como um AI, posso explicar que Python é uma linguagem de programação de alto nível. 
    Python Python Python é interpretada e tem sintaxe simples. 
    
    
    Python é muito usado em ciência de dados e desenvolvimento web.
    
    [AI Assistant]: Baseado no meu conhecimento, Python também é popular em automação.
    """
    
    print("🔧 Testando Post-processing")
    print("-" * 50)
    
    # Criar formatter
    formatter = ResponseFormatter()
    
    # Processar resposta
    processed = formatter.format_response(
        raw_answer, 
        format_type=OutputFormat.MARKDOWN,
        include_sources=False  # Sem grounding result para este teste
    )
    
    print("📝 Resposta Original:")
    print(repr(raw_answer))
    print()
    
    print("✨ Resposta Processada:")
    print(processed.content)
    print()
    
    print(f"📊 Quality Score: {processed.quality_score:.3f}")
    print(f"📏 Redução de tamanho: {len(raw_answer)} → {len(processed.content)} chars")
    
    # Teste com diferentes formatos
    print("\n🎨 Testando diferentes formatos:")
    
    for fmt in [OutputFormat.TEXT, OutputFormat.HTML, OutputFormat.JSON]:
        print(f"\n--- {fmt.value.upper()} ---")
        result = formatter.format_response(raw_answer, format_type=fmt, include_sources=False)
        print(result.content[:200] + "..." if len(result.content) > 200 else result.content)


if __name__ == "__main__":
    main()
