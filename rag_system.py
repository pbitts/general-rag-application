import os
import hashlib
import json
from pathlib import Path
from typing import List
from collections import deque

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from config import Settings, settings

# ============================================================
# METADATA MANAGER
# ============================================================

class MetadataManager:
    """Manages document's metadata to identify changes"""
    
    def __init__(self, metadata_path: str):
        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _calculate_hash(self, path: str) -> str:
        """Calculates md5 hash"""
        path_obj = Path(path)
        hash_md5 = hashlib.md5()
        
        if path_obj.is_file():
            # hash single pdf
            with open(path_obj, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        elif path_obj.is_dir():
            # hash PDF directory
            pdf_files = sorted(path_obj.glob("*.pdf"))
            for pdf in pdf_files:
                with open(pdf, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def has_changed(self, documents_path: str, config: dict) -> bool:
        """Verifies whether docs or configs have changed"""
        current_hash = self._calculate_hash(documents_path)
        
        current_metadata = {
            "documents_hash": current_hash,
            "documents_path": str(documents_path),
            "chunk_size": config.get("chunk_size"),
            "chunk_overlap": config.get("chunk_overlap"),
            "embedding_model": config.get("embedding_model")
        }
        
        if not self.metadata_path.exists():
            self._save_metadata(current_metadata)
            return True
        
        with open(self.metadata_path, 'r') as f:
            old_metadata = json.load(f)
        
        # comparing
        changed = (
            current_metadata["documents_hash"] != old_metadata.get("documents_hash") or
            current_metadata["documents_path"] != old_metadata.get("documents_path") or
            current_metadata["chunk_size"] != old_metadata.get("chunk_size") or
            current_metadata["chunk_overlap"] != old_metadata.get("chunk_overlap") or
            current_metadata["embedding_model"] != old_metadata.get("embedding_model")
        )
        
        if changed:
            self._save_metadata(current_metadata)
        
        return changed
    
    def _save_metadata(self, metadata: dict):
        """Salva metadata"""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

# ============================================================
# MEMORY MANAHER
# ============================================================

class ConversationMemory:
    """Manager conversation memory"""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.history: deque = deque(maxlen=max_size * 2)
    
    def add_exchange(self, question: str, answer: str):
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
    
    def get_history(self) -> List:
        return list(self.history)
    
    def clear(self):
        self.history.clear()
    
    def __len__(self):
        return len(self.history) // 2

# ============================================================
# LOAD DOCS
# ============================================================

class DocumentLoader:
    
    @staticmethod
    def load_documents(path: str, verbose: bool = True) -> List:
        path_obj = Path(path)
        all_pages = []
        
        if not path_obj.exists():
            raise ValueError(f"Non-existed path: {path}")
        
        if path_obj.is_file() and path_obj.suffix == '.pdf':
            # single doc
            if verbose:
                print(f"Loading: {path_obj.name}")
            loader = PyPDFLoader(str(path_obj))
            pages = loader.load()
            all_pages.extend(pages)
            if verbose:
                print(f' "{len(pages)}" pages loaded')
        
        elif path_obj.is_dir():
            # MUltiple docs
            pdf_files = sorted(path_obj.glob("*.pdf"))
            
            if not pdf_files:
                raise ValueError(f"No PDF found at: {path}")
            
            if verbose:
                print(f'Found "{len(pdf_files)}" PDFs:')
            
            for pdf_file in pdf_files:
                if verbose:
                    print(f" name: {pdf_file.name}")
                loader = PyPDFLoader(str(pdf_file))
                pages = loader.load()
                all_pages.extend(pages)
                if verbose:
                    print(f'"{len(pages)}" pages')
        
        else:
            raise ValueError(f"Invalid path: {path}")
        
        return all_pages

# ============================================================
# RAG SYSTEM
# ============================================================

class RAGSystem:
    
    def __init__(self, config: Settings):
        self.config = config
        self.embedding_model = None
        self.vectorstore = None
        self.llm = None
        self.memory = ConversationMemory(max_size=config.CONVERSATION_MEMORY_SIZE)
        self.collection_name = "documents"
        self.metadata_manager = MetadataManager(config.METADATA_PATH)
    
    def _log(self, message: str):
        if self.config.VERBOSE:
            print(message)
    
    def _get_embedding_model(self):
        if self.embedding_model is None:
            self._log(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self.embedding_model
    
    def _get_llm(self):
        if self.llm is None:
            self._log(f"Initializing LLM: {self.config.LLM_MODEL}")
            self.llm = ChatGroq(
                api_key=self.config.GROQ_API_KEY,
                model=self.config.LLM_MODEL,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
        return self.llm
    
    def _should_reindex(self) -> bool:
        # Se forçar, sempre reindexar
        if self.config.FORCE_REINDEX:
            self._log("(FORCE_REINDEX=True)")
            return True
        
        # Verifica mudanças
        config_dict = {
            "chunk_size": self.config.CHUNK_SIZE,
            "chunk_overlap": self.config.CHUNK_OVERLAP,
            "embedding_model": self.config.EMBEDDING_MODEL
        }
        
        has_changed = self.metadata_manager.has_changed(
            self.config.DOCUMENTS_PATH,
            config_dict
        )
        
        if has_changed:
            self._log("Changes detected on docs or configs")
            return True
        
        #  Verifies if Index Exists
        index_path = Path(self.config.CHROMA_DB_PATH) / self.collection_name
        if not index_path.exists():
            self._log("Index does not exist, creating...")
            return True
        
        return False
    
    def index_documents(self):
        """Index docs"""
        should_reindex = self._should_reindex()
        
        if not should_reindex:
            self._log("Loading valid index...")
            self._load_existing_index()
            return
        
        # Re-Index
        self._log("\n" + "=" * 60)
        self._log("Docs Indexing")
        self._log("=" * 60)
        
        # Load docs
        pages = DocumentLoader.load_documents(
            self.config.DOCUMENTS_PATH,
            verbose=self.config.VERBOSE
        )
        self._log(f"\n Total: {len(pages)} pages loaded")
        
        # split in chunks
        self._log(f"\n Splitting in chunks...")
        self._log(f"   - Chunk size: {self.config.CHUNK_SIZE}")
        self._log(f"   - Overlap: {self.config.CHUNK_OVERLAP}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
            separators=self.config.SEPARATORS
        )
        chunks = text_splitter.split_documents(pages)
        self._log(f' "{len(chunks)}" chunks')
        
        # Remove old index
        index_path = Path(self.config.CHROMA_DB_PATH) / self.collection_name
        if index_path.exists():
            self._log(f"\n Removing old index...")
            import shutil
            shutil.rmtree(index_path)
        
        # create index
        self._log(f"\n Creating vetorial Index...")
        embedding_model = self._get_embedding_model()
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            collection_name=self.collection_name,
            embedding=embedding_model,
            persist_directory=self.config.CHROMA_DB_PATH
        )
        
        self._log("  Indexing completed!")
        self._log("=" * 60 + "\n")
    
    def _load_existing_index(self):
        """load existing index"""
        embedding_model = self._get_embedding_model()
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=embedding_model,
            persist_directory=self.config.CHROMA_DB_PATH
        )
    
    def _get_retriever(self):
        search_kwargs = {"k": self.config.SEARCH_K}
        
        if self.config.SEARCH_TYPE == "mmr":
            search_kwargs.update({
                "fetch_k": self.config.FETCH_K,
                "lambda_mult": self.config.LAMBDA_MULT
            })
        elif self.config.SEARCH_TYPE == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.config.SCORE_THRESHOLD
        
        return self.vectorstore.as_retriever(
            search_type=self.config.SEARCH_TYPE,
            search_kwargs=search_kwargs
        )
    
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def setup_rag_chain(self):
        llm = self._get_llm()
        retriever = self._get_retriever()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant specialized in document analysis.

GUIDELINES:

Respond based ONLY on the provided context
If you cannot find the information, state it clearly
Be precise, objective, and well-structured
Use examples from the context when appropriate
Consider the history to ensure continuity

CONTEXT:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        rag_chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.memory.get_history()
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, retriever
    
    def process_questions(self):
        if not self.config.QUESTIONS:
            print("No questions found")
            return
        
        self._log("\n" + "=" * 60)
        self._log("PROCESSING QUESTIONS")
        self._log("=" * 60)
        
        rag_chain, retriever = self.setup_rag_chain()
        
        for i, question in enumerate(self.config.QUESTIONS, 1):
            print("\n" + "=" * 60)
            print(f"Question {i}/{len(self.config.QUESTIONS)}")
            print("=" * 60)
            print(f"❓ {question}")
            print("-" * 60)
            
            try:
                if self.config.SHOW_RETRIEVAL_DOCS:
                    self._log("Searching for relevant docs...")
                
                docs = retriever.invoke(question)
                
                if self.config.SHOW_RETRIEVAL_DOCS:
                    self._log(f'"{len(docs)}" Docs found\n')
                
                if self.config.VERBOSE:
                    self._log("Generating answer...\n")
                
                response = rag_chain.invoke(question)
                
                print(f"ANswer:\n{response}")
                
                self.memory.add_exchange(question, response)
                
                if self.config.SHOW_SOURCES and docs:
                    print("\n" + "-" * 60)
                    print("Sources:")
                    for j, doc in enumerate(docs, 1):
                        page = doc.metadata.get('page', 'N/A')
                        source = doc.metadata.get('source', 'N/A')
                        source_name = Path(source).name if source != 'N/A' else 'N/A'
                        
                        print(f"\n[{j}] Page {page} - {source_name}")
                        preview = doc.page_content[:200].replace('\n', ' ')
                        print(f"    {preview}...")
                
                print("=" * 60)
                
            except Exception as e:
                print(f"\n Error when processing question: {e}")
                import traceback
                traceback.print_exc()
    
    def show_summary(self):
        print("\n" + "=" * 60)
        print(" Execution Summary")
        print("=" * 60)
        print(f"📄 Docs: {self.config.DOCUMENTS_PATH}")
        print(f"🤖 Model: {self.config.LLM_MODEL}")
        print(f"🔍 Search: {self.config.SEARCH_TYPE} (K={self.config.SEARCH_K})")
        print(f"✂️  Chunks: {self.config.CHUNK_SIZE} chars (overlap={self.config.CHUNK_OVERLAP})")
        print(f"❓ Processed Questions: {len(self.config.QUESTIONS)}")
        print(f"🧠 Conversations in Memory: {len(self.memory)}")
        print("=" * 60 + "\n")
