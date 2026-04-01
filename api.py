from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import copy

from config import settings
from rag_system import RAGSystem
from observability import configure_langsmith

configure_langsmith()
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

# Estado global simples (demo local, single-user)
rag: Optional[RAGSystem] = None
rag_chain = None
retriever = None

class IndexRequest(BaseModel):
    documents_path: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_type: str = "mmr"
    search_k: int = 4
    force_reindex: bool = False
    temperature: float = 0.0

class ChatRequest(BaseModel):
    question: str
    show_sources: bool = True

@app.get("/")
def root():
    return FileResponse("ui/index.html")

@app.post("/index")
def index_documents(req: IndexRequest):
    global rag, rag_chain, retriever

    cfg = copy.deepcopy(settings)
    cfg.update_from_dict(req.model_dump())

    rag = RAGSystem(cfg)
    rag.index_documents()
    rag_chain, retriever = rag.setup_rag_chain()

    return {"status": "ok", "message": "Indexing completed."}

@app.post("/chat")
def chat(req: ChatRequest):
    if rag is None or rag_chain is None:
        return {"error": "Index documents first."}

    docs = retriever.invoke(req.question)
    response = rag_chain.invoke(req.question)
    rag.memory.add_exchange(req.question, response)

    sources = []
    if req.show_sources:
        for doc in docs:
            sources.append({
                "page":    doc.metadata.get("page", "N/A"),
                "source":  doc.metadata.get("source", "N/A").split("/")[-1],
                "preview": doc.page_content[:200].replace("\n", " ")
            })

    return {"answer": response, "sources": sources}

@app.post("/clear-memory")
def clear_memory():
    if rag:
        rag.memory.clear()
    return {"status": "ok"}

@app.get("/status")
def status():
    return {
        "indexed": rag is not None,
        "memory_turns": len(rag.memory) if rag else 0
    }