from rag_system import RAGSystem
from config import Settings, settings
from observability import configure_langsmith

def main():
    print("\n" + "=" * 60)
    print("RAG SISTEM - AUTOMATIC CHANGES DETECTION")
    print("=" * 60)
    
    rag_system = RAGSystem(settings)
    
    rag_system.index_documents()
    
    rag_system.process_questions()
    
    rag_system.show_summary()

if __name__ == "__main__":
    configure_langsmith()
    main()