

import os
from data_ingestion import document_ingestion_orchestrator
from dotenv import load_dotenv
# Load environment variables from .env file if available
load_dotenv()

if __name__ == "__main__":
    adapter = document_ingestion_orchestrator.DocumentIngestionOrchestrator()
    
    # Check if Azure credentials are provided via environment variables
    storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    storage_account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME", "rag-agents-container")
    
    if storage_account_name and storage_account_key:
        # Use Azure Blob Storage if credentials are available
        print("Azure credentials found. Processing files from Azure Blob Storage...")
        document_ingestion_orchestrator.ingest_azure_blob_container(
            adapter, storage_account_name, storage_account_key, container_name
        )
    else:
        # Fall back to local directory processing
        print("No Azure credentials found. Processing local directory...")
        document_ingestion_orchestrator.ingest_directory(adapter, "doc-ingestion")
