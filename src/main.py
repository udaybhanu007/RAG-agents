

import os
from data_ingestion import document_ingestion_orchestrator
from core.azure_keyvault_manager import get_secret_from_keyvault

def run_document_ingestion():
    """
    Run document ingestion process from Azure Blob Storage or local directory
    
    Returns:
        bool: True if ingestion was successful, False otherwise
    """
    try:
        storage_account_name = get_secret_from_keyvault("AZURE_STORAGE_ACCOUNT_NAME")
        storage_account_key = get_secret_from_keyvault("AZURE_STORAGE_ACCOUNT_KEY")
        container_name = get_secret_from_keyvault("AZURE_BLOB_CONTAINER_NAME") or "rag-agents-container"
        adapter = document_ingestion_orchestrator.DocumentIngestionOrchestrator()
        
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
        
        return True
        
    except Exception as e:
        print(f"Error during document ingestion: {e}")
        return False

# def run_workflow_query(query: str) -> str:
#     """
#     Run a query through the Multi-Agent RAG Workflow
    
#     Args:
#         query (str): The user's query
        
#     Returns:
#         str: The workflow response
#     """
#     try:
#         from agents.multi_agent_rag_workflow import MultiAgentRAGWorkflow
        
#         # Initialize workflow
#         workflow = MultiAgentRAGWorkflow()
        
#         # Run the query
#         result = workflow.run(query)
        
#         return result
        
#     except Exception as e:
#         return f"Error processing query: {e}"

if __name__ == "__main__":
    # Run document ingestion when script is executed directly
    success = run_document_ingestion()
    
    if success:
        print("✅ Document ingestion completed successfully!")
        
        # # Example query to test the workflow
        # test_query = "What is NIH Chest X-ray?"
        # print(f"\n🧠 Testing workflow with query: {test_query}")
        # result = run_workflow_query(test_query)
        # print(f"Result: {result}")
    else:
        print("❌ Document ingestion failed!")
