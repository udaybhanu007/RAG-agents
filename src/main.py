

import os
from data_ingestion import document_ingestion_orchestrator
from core.azure_keyvault_manager import get_secret_from_keyvault

if __name__ == "__main__":

    storage_account_name = get_secret_from_keyvault("AZURE_STORAGE_ACCOUNT_NAME")
    storage_account_key = get_secret_from_keyvault("AZURE_STORAGE_ACCOUNT_KEY")
    container_name = get_secret_from_keyvault("AZURE_BLOB_CONTAINER_NAME") or "rag-agents-container"
    adapter = document_ingestion_orchestrator.DocumentIngestionOrchestrator()
    
    # Get Azure credentials from Key Vault
   

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
