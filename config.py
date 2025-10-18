"""
Configuração central do sistema RAG
"""
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Config(BaseModel):
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    INDEX_DIR: Path = PROJECT_ROOT / "indices"
    
    # Database configs
    SQLITE_DB_PATH: Path = DATA_DIR / "documents.db"
    
    # Index configs
    WHOOSH_INDEX_DIR: Path = INDEX_DIR / "whoosh"
    FAISS_INDEX_PATH: Path = INDEX_DIR / "faiss_index"
    
    # Model configs
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    
    # LLM configs
    GROQ_API_KEY: str = "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"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    
    # Search configs
    TOP_K_LEXICAL: int = 20
    TOP_K_VECTOR: int = 20
    TOP_K_FINAL: int = 10
    
    # Rerank weights
    LEXICAL_WEIGHT: float = 0.4
    VECTOR_WEIGHT: float = 0.6
    
    # Grounding configs
    MIN_CONFIDENCE_THRESHOLD: float = 0.7
    MAX_PASSAGE_LENGTH: int = 500
    
    class Config:
        env_file = ".env"

# Instância global da configuração
config = Config()

# Criar diretórios necessários
def setup_directories():
    """Cria os diretórios necessários se não existirem"""
    config.DATA_DIR.mkdir(exist_ok=True)
    config.INDEX_DIR.mkdir(exist_ok=True)
    config.WHOOSH_INDEX_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    setup_directories()
    print("Diretórios criados com sucesso!")