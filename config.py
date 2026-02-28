from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ===============================
    # Model
    # ===============================
    GROQ_API_KEY: str 

    # ===============================
    # Systems Config
    # ===============================
    
    # DOCUMENT PATH
    # SINGLE PDF DOCUMENT OR DIRECTORY WITH MANY PDFS
    DOCUMENTS_PATH: str
    # DOCUMENTS_PATH: str = "./documents"
    
    # DIRECTORIES
    CHROMA_DB_PATH: str = "./chroma_db"
    METADATA_PATH: str = "./chroma_db/.metadata.json"  # Stores documents Hashes
        
    # FOUNDATION MODELS
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "openai/gpt-oss-120b"

    # LLM CONFIGS
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 2000
    
    # TEXT SPLITTER
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]
    
    # RETRIEVAL
    SEARCH_TYPE: str = "similarity"
    SEARCH_K: int = 4
    FETCH_K: int = 20
    LAMBDA_MULT: float = 0.5
    SCORE_THRESHOLD: float = 0.5
    
    # MEMORY
    CONVERSATION_MEMORY_SIZE: int = 5
    
    # QUESTIONS
    QUESTIONS: List[str] = [
        "What is RLHF?",
        "What is SFT?"
    ]
    
    # PREFERENCES
    SHOW_SOURCES: bool = True
    SHOW_RETRIEVAL_DOCS: bool = True
    VERBOSE: bool = True
    
    # RE-INDEX
    FORCE_REINDEX: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()