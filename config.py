from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ===============================
    # Model
    # ===============================
    GROQ_API_KEY: str 

    # ===============================
    # Tracing
    # ===============================
    LANGSMITH_TRACING: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

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
        frozen = False

    def update_from_dict(self, data: dict):
        if "documents_path"   in data: self.DOCUMENTS_PATH   = data["documents_path"]
        if "chunk_size"       in data: self.CHUNK_SIZE        = int(data["chunk_size"])
        if "chunk_overlap"    in data: self.CHUNK_OVERLAP     = int(data["chunk_overlap"])
        if "search_type"      in data: self.SEARCH_TYPE       = data["search_type"]
        if "search_k"         in data: self.SEARCH_K          = int(data["search_k"])
        if "llm_model"        in data: self.LLM_MODEL         = data["llm_model"]
        if "force_reindex"    in data: self.FORCE_REINDEX     = bool(data["force_reindex"])
        if "temperature"      in data: self.TEMPERATURE       = float(data["temperature"])

settings = Settings()