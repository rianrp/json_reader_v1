"""
Módulo de Grounding
Responsável por:
- Cortar passagens em tamanhos adequados
- Validar se a resposta gerada está contida no contexto
- String matching e cosine similarity checking
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import difflib

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config


@dataclass
class Passage:
    """Representa uma passagem de texto"""
    text: str
    source_doc_id: str
    source_title: str
    start_pos: int
    end_pos: int
    score: Optional[float] = None


@dataclass
class GroundingResult:
    """Resultado da validação de grounding"""
    is_grounded: bool
    confidence_score: float
    supporting_passages: List[Passage]
    string_matches: List[Dict[str, Any]]
    cosine_similarities: List[float]
    metadata: Dict[str, Any]


class PassageCutter:
    """Responsável por cortar documentos em passagens menores"""
    
    def __init__(self, max_passage_length: int = None):
        self.max_passage_length = max_passage_length or config.MAX_PASSAGE_LENGTH
    
    def cut_by_sentences(self, text: str, doc_id: str, title: str) -> List[Passage]:
        """
        Corta texto em passagens baseadas em sentenças
        
        Args:
            text: Texto a ser cortado
            doc_id: ID do documento fonte
            title: Título do documento
            
        Returns:
            Lista de Passage
        """
        passages = []
        
        # Dividir em sentenças (regex simples)
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text)
        
        current_passage = ""
        start_pos = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Verificar se adicionar a sentença ultrapassa o limite
            potential_passage = current_passage + " " + sentence if current_passage else sentence
            
            if len(potential_passage) <= self.max_passage_length:
                current_passage = potential_passage
            else:
                # Salvar passagem atual se não estiver vazia
                if current_passage:
                    end_pos = start_pos + len(current_passage)
                    passage = Passage(
                        text=current_passage.strip(),
                        source_doc_id=doc_id,
                        source_title=title,
                        start_pos=start_pos,
                        end_pos=end_pos
                    )
                    passages.append(passage)
                    start_pos = end_pos
                
                # Iniciar nova passagem
                current_passage = sentence
        
        # Adicionar última passagem
        if current_passage:
            end_pos = start_pos + len(current_passage)
            passage = Passage(
                text=current_passage.strip(),
                source_doc_id=doc_id,
                source_title=title,
                start_pos=start_pos,
                end_pos=end_pos
            )
            passages.append(passage)
        
        return passages
    
    def cut_by_sliding_window(
        self, 
        text: str, 
        doc_id: str, 
        title: str,
        window_size: int = None,
        overlap: int = 50
    ) -> List[Passage]:
        """
        Corta texto usando janela deslizante
        
        Args:
            text: Texto a ser cortado
            doc_id: ID do documento
            title: Título do documento
            window_size: Tamanho da janela (chars)
            overlap: Sobreposição entre janelas
            
        Returns:
            Lista de Passage
        """
        if window_size is None:
            window_size = self.max_passage_length
        
        passages = []
        text_length = len(text)
        
        start = 0
        while start < text_length:
            end = min(start + window_size, text_length)
            
            # Tentar terminar em uma quebra natural (espaço, ponto)
            if end < text_length:
                for i in range(end, max(end - 50, start), -1):
                    if text[i] in ' .\n':
                        end = i
                        break
            
            passage_text = text[start:end].strip()
            
            if passage_text:
                passage = Passage(
                    text=passage_text,
                    source_doc_id=doc_id,
                    source_title=title,
                    start_pos=start,
                    end_pos=end
                )
                passages.append(passage)
            
            # Mover janela com sobreposição
            start = max(start + window_size - overlap, end)
        
        return passages
    
    def cut_documents(self, documents: List[Dict[str, Any]], method: str = "sentences") -> List[Passage]:
        """
        Corta múltiplos documentos em passagens
        
        Args:
            documents: Lista de documentos
            method: Método de corte ("sentences" ou "sliding")
            
        Returns:
            Lista de todas as passagens
        """
        all_passages = []
        
        for doc in documents:
            doc_id = doc.get('id', 'unknown')
            title = doc.get('title', 'Untitled')
            content = doc.get('content', '')
            
            if method == "sentences":
                passages = self.cut_by_sentences(content, doc_id, title)
            elif method == "sliding":
                passages = self.cut_by_sliding_window(content, doc_id, title)
            else:
                raise ValueError(f"Método desconhecido: {method}")
            
            all_passages.extend(passages)
        
        logger.info(f"Cortados {len(documents)} documentos em {len(all_passages)} passagens")
        return all_passages


class StringMatcher:
    """Responsável por matching de strings"""
    
    def __init__(self):
        pass
    
    def exact_match(self, text1: str, text2: str) -> float:
        """Match exato entre textos"""
        return 1.0 if text1.lower().strip() == text2.lower().strip() else 0.0
    
    def substring_match(self, needle: str, haystack: str) -> float:
        """Verifica se needle está contido em haystack"""
        needle_clean = needle.lower().strip()
        haystack_clean = haystack.lower().strip()
        return 1.0 if needle_clean in haystack_clean else 0.0
    
    def fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> float:
        """Match fuzzy usando SequenceMatcher"""
        similarity = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity if similarity >= threshold else 0.0
    
    def token_overlap(self, text1: str, text2: str) -> float:
        """Calcula sobreposição de tokens"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def find_matches(
        self, 
        generated_text: str, 
        passages: List[Passage]
    ) -> List[Dict[str, Any]]:
        """
        Encontra matches entre texto gerado e passagens
        
        Args:
            generated_text: Texto gerado pelo LLM
            passages: Lista de passagens para verificar
            
        Returns:
            Lista de matches encontrados
        """
        matches = []
        
        # Dividir texto gerado em sentenças para análise granular
        sentences = re.split(r'[.!?]+', generated_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for sentence in sentences:
            for passage in passages:
                # Testar diferentes tipos de match
                exact = self.exact_match(sentence, passage.text)
                substring = self.substring_match(sentence, passage.text)
                fuzzy = self.fuzzy_match(sentence, passage.text)
                token_overlap = self.token_overlap(sentence, passage.text)
                
                # Se houver algum match significativo
                if any(score > 0.5 for score in [exact, substring, fuzzy, token_overlap]):
                    match = {
                        'sentence': sentence,
                        'passage': passage,
                        'scores': {
                            'exact': exact,
                            'substring': substring,
                            'fuzzy': fuzzy,
                            'token_overlap': token_overlap
                        },
                        'best_score': max(exact, substring, fuzzy, token_overlap)
                    }
                    matches.append(match)
        
        # Ordenar por melhor score
        matches.sort(key=lambda x: x['best_score'], reverse=True)
        return matches


class SemanticMatcher:
    """Responsável por matching semântico usando embeddings"""
    
    def __init__(self):
        self.model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
    
    def cosine_similarity_check(
        self, 
        generated_text: str, 
        passages: List[Passage],
        threshold: float = 0.7
    ) -> Tuple[List[float], List[Passage]]:
        """
        Verifica similaridade semântica entre texto gerado e passagens
        
        Args:
            generated_text: Texto gerado
            passages: Passagens para comparar
            threshold: Threshold de similaridade
            
        Returns:
            Tuple com (similaridades, passagens_relevantes)
        """
        if not passages:
            return [], []
        
        # Gerar embeddings
        texts = [generated_text] + [p.text for p in passages]
        embeddings = self.model.encode(texts)
        
        # Calcular similaridades
        generated_embedding = embeddings[0:1]
        passage_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(generated_embedding, passage_embeddings)[0]
        
        # Filtrar passagens relevantes
        relevant_passages = []
        relevant_similarities = []
        
        for similarity, passage in zip(similarities, passages):
            if similarity >= threshold:
                relevant_similarities.append(float(similarity))
                relevant_passages.append(passage)
        
        return relevant_similarities, relevant_passages


class GroundingValidator:
    """Classe principal para validação de grounding"""
    
    def __init__(self):
        self.passage_cutter = PassageCutter()
        self.string_matcher = StringMatcher()
        self.semantic_matcher = SemanticMatcher()
        self.confidence_threshold = config.MIN_CONFIDENCE_THRESHOLD
    
    def validate_grounding(
        self, 
        generated_answer: str, 
        context_documents: List[Dict[str, Any]],
        use_passages: bool = True
    ) -> GroundingResult:
        """
        Valida se a resposta gerada está fundamentada no contexto
        
        Args:
            generated_answer: Resposta gerada pelo LLM
            context_documents: Documentos usados como contexto
            use_passages: Se deve cortar em passagens menores
            
        Returns:
            GroundingResult com validação completa
        """
        try:
            # Preparar passagens
            if use_passages:
                passages = self.passage_cutter.cut_documents(context_documents)
            else:
                passages = []
                for doc in context_documents:
                    passage = Passage(
                        text=doc.get('content', ''),
                        source_doc_id=doc.get('id', 'unknown'),
                        source_title=doc.get('title', 'Untitled'),
                        start_pos=0,
                        end_pos=len(doc.get('content', ''))
                    )
                    passages.append(passage)
            
            # String matching
            string_matches = self.string_matcher.find_matches(generated_answer, passages)
            
            # Semantic matching
            cosine_similarities, supporting_passages = self.semantic_matcher.cosine_similarity_check(
                generated_answer, passages
            )
            
            # Calcular score de confiança
            confidence_score = self._calculate_confidence(
                string_matches, cosine_similarities, generated_answer, passages
            )
            
            # Determinar se está fundamentado
            is_grounded = confidence_score >= self.confidence_threshold
            
            # Metadata adicional
            metadata = {
                'total_passages': len(passages),
                'string_matches_count': len(string_matches),
                'semantic_matches_count': len(supporting_passages),
                'avg_cosine_similarity': np.mean(cosine_similarities) if cosine_similarities else 0.0,
                'max_cosine_similarity': np.max(cosine_similarities) if cosine_similarities else 0.0,
                'answer_length': len(generated_answer),
                'context_documents_count': len(context_documents)
            }
            
            result = GroundingResult(
                is_grounded=is_grounded,
                confidence_score=confidence_score,
                supporting_passages=supporting_passages,
                string_matches=string_matches,
                cosine_similarities=cosine_similarities,
                metadata=metadata
            )
            
            logger.info(
                f"Grounding validation: {'✅ GROUNDED' if is_grounded else '❌ NOT GROUNDED'} "
                f"(confidence: {confidence_score:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na validação de grounding: {e}")
            return GroundingResult(
                is_grounded=False,
                confidence_score=0.0,
                supporting_passages=[],
                string_matches=[],
                cosine_similarities=[],
                metadata={'error': str(e)}
            )
    
    def _calculate_confidence(
        self, 
        string_matches: List[Dict[str, Any]], 
        cosine_similarities: List[float],
        generated_answer: str,
        passages: List[Passage]
    ) -> float:
        """
        Calcula score de confiança baseado em múltiplos fatores
        
        Returns:
            Score de confiança entre 0 e 1
        """
        
        # Componente 1: String matching
        string_score = 0.0
        if string_matches:
            # Média dos melhores scores de string matching
            string_score = np.mean([match['best_score'] for match in string_matches])
        
        # Componente 2: Similaridade semântica
        semantic_score = 0.0
        if cosine_similarities:
            # Usar máximo e média das similaridades
            semantic_score = 0.7 * np.max(cosine_similarities) + 0.3 * np.mean(cosine_similarities)
        
        # Componente 3: Cobertura (quantas sentenças da resposta têm suporte)
        coverage_score = 0.0
        answer_sentences = re.split(r'[.!?]+', generated_answer)
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]
        
        if answer_sentences:
            supported_sentences = 0
            for sentence in answer_sentences:
                # Verificar se a sentença tem suporte (string ou semântico)
                has_string_support = any(
                    match['sentence'].lower() in sentence.lower() 
                    for match in string_matches
                )
                
                if has_string_support:
                    supported_sentences += 1
                else:
                    # Verificar suporte semântico
                    sentence_embedding = self.semantic_matcher.model.encode([sentence])
                    for passage in passages:
                        passage_embedding = self.semantic_matcher.model.encode([passage.text])
                        similarity = cosine_similarity(sentence_embedding, passage_embedding)[0][0]
                        if similarity > 0.6:  # Threshold mais baixo para cobertura
                            supported_sentences += 1
                            break
            
            coverage_score = supported_sentences / len(answer_sentences)
        
        # Combinar scores com pesos
        final_score = (
            0.4 * string_score +      # String matching é mais confiável
            0.4 * semantic_score +    # Semântica é importante
            0.2 * coverage_score      # Cobertura garante que toda resposta tem suporte
        )
        
        return min(final_score, 1.0)  # Cap em 1.0


def main():
    """Função principal para teste"""
    
    # Documentos de exemplo
    documents = [
        {
            'id': '1',
            'title': 'Python Basics',
            'content': 'Python é uma linguagem de programação de alto nível. É interpretada e tem sintaxe simples. Python é muito usado em ciência de dados e desenvolvimento web.'
        },
        {
            'id': '2',
            'title': 'Python Applications',
            'content': 'Python é amplamente usado em inteligência artificial, machine learning, automação e desenvolvimento de aplicações web. Sua simplicidade torna Python popular entre iniciantes.'
        }
    ]
    
    # Resposta gerada (exemplo)
    generated_answer = "Python é uma linguagem interpretada de alto nível com sintaxe simples. É muito popular para ciência de dados e inteligência artificial."
    
    # Criar validator
    validator = GroundingValidator()
    
    print("🔍 Testando Grounding Validation")
    print("-" * 50)
    
    # Validar grounding
    result = validator.validate_grounding(generated_answer, documents)
    
    print(f"✅ Grounded: {result.is_grounded}")
    print(f"📊 Confidence: {result.confidence_score:.3f}")
    print(f"🔤 String matches: {len(result.string_matches)}")
    print(f"🧠 Semantic matches: {len(result.supporting_passages)}")
    
    if result.cosine_similarities:
        print(f"📈 Avg cosine similarity: {np.mean(result.cosine_similarities):.3f}")
        print(f"📈 Max cosine similarity: {np.max(result.cosine_similarities):.3f}")
    
    print(f"📝 Metadata: {result.metadata}")
    
    # Mostrar algumas passagens de suporte
    if result.supporting_passages:
        print("\n🎯 Passagens de suporte:")
        for i, passage in enumerate(result.supporting_passages[:3], 1):
            print(f"{i}. [{passage.source_title}]: {passage.text[:100]}...")


if __name__ == "__main__":
    main()
