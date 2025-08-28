
import ssl
import urllib3
import os

# Disable SSL verification globally before any other imports
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_VERIFY'] = 'false'
os.environ['PYTHONHTTPSVERIFY'] = '0'

from data_ingestion import document_ingestion_orchestrator
from core.azure_keyvault_manager import get_secret_from_keyvault

def run_document_ingestion():
    """
    Run document ingestion process from Azure Blob Storage, Confluence pages, Box folder, or local directory
    
    Returns:
        bool: True if ingestion was successful, False otherwise
    """
    try:
        storage_account_name = get_secret_from_keyvault("AZURE_STORAGE_ACCOUNT_NAME")
        storage_account_key = get_secret_from_keyvault("AZURE_STORAGE_ACCOUNT_KEY")
        container_name = get_secret_from_keyvault("AZURE_BLOB_CONTAINER_NAME") or "rag-agents-container"
        confluence_pages  = ["MPC-POC", "User Story-Requirement"]
        box_folder_name = get_secret_from_keyvault("BOX_FOLDER_NAME") or "documents-ingest"
        adapter = document_ingestion_orchestrator.DocumentIngestionOrchestrator()
        
        if storage_account_name and storage_account_key:
            # Use Azure Blob Storage if credentials are available
            print("Azure credentials found. Downloading files from Azure Blob Storage...")
            
            # Step 1: Download files from Azure Blob Storage
            download_result = adapter.download_azure_blob(
                storage_account_name, storage_account_key, container_name
            )
            
            if download_result["downloaded_count"] > 0:
                print(f"Azure download completed: {download_result['downloaded_count']} files downloaded")
            else:
                print("No files were downloaded from Azure Blob Storage.")
            
            # Step 2: Download Confluence pages
            if confluence_pages:
                print("Downloading Confluence pages...")
                page_titles = confluence_pages
                
                confluence_result = adapter.download_from_confluence_pages(page_titles)
                
                if confluence_result["downloaded_count"] > 0:
                    print(f"Confluence download completed: {confluence_result['downloaded_count']} pages downloaded")
                else:
                    print("No Confluence pages were downloaded.")
            else:
                print("No Confluence page titles configured.")
            
            # Step 3: Download Box files
            if box_folder_name:
                print("Downloading Box files...")
                
                box_result = adapter.download_from_box(box_folder_name)
                
                if box_result["downloaded_count"] > 0:
                    print(f"Box download completed: {box_result['downloaded_count']} files downloaded")
                else:
                    print("No Box files were downloaded.")
            else:
                print("No Box folder name configured.")
            
            # Step 4: Process all downloaded files at once
            print("\nProcessing all downloaded files...")
            download_dir = download_result["download_dir"]  # All sources use the same directory
            
            final_process_result = adapter.process_downloaded_folder(download_dir, cleanup=False)
            print(f"Final processing completed: {final_process_result['processed_count']} files processed, {final_process_result['error_count']} errors")
        else:
            # Fall back to local directory processing
            print("No Azure credentials found. Processing local directory...")
            document_ingestion_orchestrator.ingest_directory(adapter, "doc-ingestion")
        
        return True
        
    except Exception as e:
        print(f"Error during document ingestion: {e}")
        return False

if __name__ == "__main__":
    # Run document ingestion when script is executed directly
    success = run_document_ingestion()
    
    if success:
        print("✅ Document ingestion completed successfully!")
    else:
        print("❌ Document ingestion failed!")
