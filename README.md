# Sistema RAG Completo em Python

Um sistema RAG (Retrieval-Augmented Generation) completo implementado em Python, seguindo a arquitetura modular recomendada.

## 🏗️ Arquitetura

O sistema é composto por 7 módulos principais, construídos na ordem ideal:

### 1️⃣ **index/** - Ingestão e Indexação
- Ingestão de documentos JSON
- Criação de índice léxico (Whoosh)
- Criação de índice vetorial (FAISS)
- Armazenamento em SQLite

### 2️⃣ **lexical_vector/** - Busca Híbrida
- `search_lexical(query)` - Busca léxica com Whoosh
- `search_vector(query)` - Busca vetorial com FAISS
- `hybrid_merge()` - Combinação inteligente dos resultados

### 3️⃣ **rerank/** - Reranqueamento
- BM25 scoring
- Similaridade de cosseno
- Heurísticas customizadas
- Opção de ML para ranqueamento

### 4️⃣ **llm/** - Interface LLM
- Suporte para Groq (principal)
- Fallback para OpenAI e HuggingFace
- Interface unificada `generate_answer(prompt)`
- Templates otimizados para RAG

### 5️⃣ **grounding/** - Validação de Fundamentação
- Corte inteligente de passagens
- Validação string matching
- Verificação de similaridade semântica
- Score de confiança

### 6️⃣ **post-processing/** - Formatação
- Limpeza de texto (remoção de "lixo")
- Formatação em múltiplos formatos (Markdown, HTML, JSON, Texto)
- Destaque de fontes com confiança
- Cálculo de qualidade

### 7️⃣ **telemetry-learning/** - Telemetria e Aprendizado
- Log de queries, cliques, feedbacks
- Analytics de desempenho
- Otimização automática de pesos
- Feedback loop para melhoria contínua

## 🚀 Instalação

### 1. Clonar/Baixar o projeto

```bash
git clone <repository-url>
cd json_reader_v1
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

**Dependências principais:**
- `whoosh` - Índice léxico
- `faiss-cpu` - Índice vetorial
- `sentence-transformers` - Embeddings
- `groq` - LLM API (chave: `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`)
- `rank-bm25` - Algoritmo BM25
- `scikit-learn` - Métricas ML
- `pandas` - Análise de dados
- `loguru` - Logging

### 3. Configurar ambiente (opcional)

Crie um arquivo `.env` na raiz do projeto:

```env
GROQ_API_KEY=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
GROQ_MODEL=llama3-8b-8192
```

## 🎯 Uso Rápido

### Sistema Completo

```python
from rag_system import RAGSystem, OutputFormat

# Inicializar sistema
rag = RAGSystem(enable_telemetry=True)

# Indexar documentos JSON
success = rag.index_documents("meus_documentos.json")

# Fazer query
response = rag.query(
    "Qual é a capital do Brasil?",
    top_k=5,
    output_format=OutputFormat.MARKDOWN,
    include_sources=True
)

print(f"Resposta: {response.answer}")
print(f"Confiança: {response.confidence_score}")
print(f"Fundamentado: {response.is_grounded}")
print(f"Tempo: {response.response_time_ms}ms")
```

### Uso Modular

```python
# Apenas indexação
from index.main import IndexManager
manager = IndexManager()
manager.build_indices_from_json("documentos.json")

# Apenas busca
from lexical_vector.main import HybridSearcher
searcher = HybridSearcher()
results = searcher.search("minha query", top_k=10)

# Apenas LLM
from llm.main import LLMManager
llm = LLMManager()
response = llm.generate_answer("Explique Python")
```

## 📁 Formato dos Documentos

O sistema aceita documentos JSON no formato:

```json
[
  {
    "title": "Título do Documento",
    "content": "Conteúdo completo do documento...",
    "metadata": "opcional"
  },
  {
    "title": "Outro Documento", 
    "content": "Mais conteúdo...",
    "category": "categoria",
    "tags": ["tag1", "tag2"]
  }
]
```

## 🔧 Configurações

Edite `config.py` para ajustar:

```python
# Número de resultados
TOP_K_LEXICAL = 20
TOP_K_VECTOR = 20  
TOP_K_FINAL = 10

# Pesos da busca híbrida
LEXICAL_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

# Modelo de embeddings
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# Grounding
MIN_CONFIDENCE_THRESHOLD = 0.7
MAX_PASSAGE_LENGTH = 500
```

## 📊 Analytics e Telemetria

```python
# Obter métricas
analytics = rag.get_analytics(days=7)
print(f"Total queries: {analytics['performance_metrics']['total_queries']}")
print(f"CTR: {analytics['performance_metrics']['global_ctr']:.2%}")

# Log de feedback
rag.log_user_feedback(response.query_id, FeedbackType.THUMBS_UP)
rag.log_result_click("doc_123", position=0, query="minha busca")

# Otimização automática
optimized = rag.optimize_performance()
print(f"Sistema otimizado: {optimized}")
```

## 🎨 Formatos de Saída

O sistema suporta múltiplos formatos:

```python
# Markdown (padrão)
response = rag.query("pergunta", output_format=OutputFormat.MARKDOWN)

# HTML com CSS
response = rag.query("pergunta", output_format=OutputFormat.HTML)

# JSON estruturado
response = rag.query("pergunta", output_format=OutputFormat.JSON)

# Texto simples
response = rag.query("pergunta", output_format=OutputFormat.TEXT)
```

## 🧪 Executar Demonstração

```bash
python rag_system.py
```

Isso irá:
1. Criar documentos de exemplo
2. Indexar os documentos
3. Executar queries de teste
4. Mostrar analytics
5. Testar otimização automática

## 🔍 Testando Módulos Individualmente

```bash
# Testar indexação
python index/main.py

# Testar busca híbrida  
python lexical_vector/main.py

# Testar reranking
python rerank/main.py

# Testar LLM
python llm/main.py

# Testar grounding
python grounding/main.py

# Testar pós-processamento
python post-processing/main.py

# Testar telemetria
python telemetry-learning/main.py
```

## 🛠️ Estrutura de Arquivos

```
json_reader_v1/
├── config.py                 # Configurações centrais
├── requirements.txt          # Dependências
├── rag_system.py            # Sistema principal
├── README.md                # Este arquivo
├── data/                    # Dados gerados
│   ├── documents.db        # SQLite com documentos
│   └── telemetry.db        # SQLite com telemetria
├── indices/                 # Índices gerados
│   ├── whoosh/             # Índice léxico
│   └── faiss_index         # Índice vetorial
├── index/
│   └── main.py
├── lexical_vector/
│   └── main.py  
├── rerank/
│   └── main.py
├── llm/
│   └── main.py
├── grounding/
│   └── main.py
├── post-processing/
│   └── main.py
└── telemetry-learning/
    └── main.py
```

## 🚨 Solução de Problemas

### Erro de imports
```bash
# Instalar dependências faltantes
pip install sentence-transformers faiss-cpu whoosh groq loguru pandas scikit-learn rank-bm25
```

### Erro de API key
- Verificar se `GROQ_API_KEY` está configurada
- Testar conexão: `curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/openai/v1/models`

### Performance lenta
- Reduzir `TOP_K_LEXICAL` e `TOP_K_VECTOR` em `config.py`
- Usar modelo de embedding menor: `"all-MiniLM-L6-v2"`
- Otimizar tamanho dos documentos

### Resultados ruins
- Verificar qualidade dos documentos indexados
- Ajustar pesos em `config.py`
- Executar otimização automática: `rag.optimize_performance()`

## 🤝 Contribuição

1. Cada módulo é independente e testável
2. Seguir padrões de logging com `loguru`
3. Documentar funções com docstrings
4. Adicionar testes em `main()` de cada módulo
5. Manter retrocompatibilidade na API

## 📝 Licença

MIT License - Livre para uso e modificação.

---

**Desenvolvido seguindo as melhores práticas de arquitetura RAG modular!** 🚀