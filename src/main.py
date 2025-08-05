

from data_ingestion import document_ingestion_orchestrator

if __name__ == "__main__":
    adapter = document_ingestion_orchestrator.DocumentIngestionOrchestrator()
    # Change 'docs' to your target directory as needed
    document_ingestion_orchestrator.ingest_directory(adapter, "doc-ingestion")
