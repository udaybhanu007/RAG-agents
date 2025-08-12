from .classify_document import classify_document, analyze_repository_documents
from .mixed_document import MixedDocumentIngestor
from .ingestion_structured_document import StructuredDocumentIngestor
from .ingestion_unstructured_document import UnstructuredDocumentIngestor
import os
import tempfile
from typing import TYPE_CHECKING

# Optional Azure Blob Storage import
if TYPE_CHECKING:
    from azure.storage.blob import BlobServiceClient

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

class DocumentIngestionOrchestrator:
    """
    Adapter class to unify ingestion for mixed, structured, and unstructured documents.
    """
    def __init__(self, mixed_ingestor=None, structured_ingestor=None, unstructured_ingestor=None):
        self.mixed_ingestor = mixed_ingestor or MixedDocumentIngestor()
        self.structured_ingestor = structured_ingestor or StructuredDocumentIngestor()
        self.unstructured_ingestor = unstructured_ingestor or UnstructuredDocumentIngestor()

    def process_document(self, file_path: str) -> dict:
        """
        Classifies the document and ingests it based on its type.
        """
        doc_type, content = classify_document(file_path)
        print(f"Document type for {file_path}: {doc_type}")
        
        if doc_type == "mixed":
            ingestion_result = self.mixed_ingestor.ingest_mixed_document(file_path, content)
        elif doc_type == "structured":
            ingestion_result = self.structured_ingestor.ingest_structured_document(file_path, content) # type: ignore
        elif doc_type == "unstructured":
            ingestion_result = self.unstructured_ingestor.ingest_unstructured_document(file_path, content) # type: ignore
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        return {"classification": doc_type, "content": content, "ingestion_result": ingestion_result}

    def analyze_directory(self, directory_path: str) -> dict:
        """
        Analyzes all documents in the given directory and classifies them.       
        """
        return analyze_repository_documents(directory_path)

    def ingest_from_azure_blob(self, storage_account_name: str, storage_account_key: str, container_name: str = "rag-agents-container"):
        """Downloads and processes all files from Azure Blob Storage container."""
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Blob Storage is not available. Please install azure-storage-blob: pip install azure-storage-blob")
        
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=storage_account_key
        )
        
        container_client = blob_service_client.get_container_client(container_name)
        print(f"Processing files from Azure Blob Storage container: {container_name}")
        
        processed_count = error_count = 0
        errors_log = []
        
        for blob in container_client.list_blobs():
            print(f"\nProcessing blob: {blob.name}")
            temp_file_path = None
            
            try:
                # Create temp directory if it doesn't exist
                temp_dir = tempfile.gettempdir()
                blob_filename = os.path.basename(blob.name)
                temp_file_path = os.path.join(temp_dir, blob_filename)
                
                # Download blob to temporary file with actual name
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
                with open(temp_file_path, "wb") as download_file:
                    download_data = blob_client.download_blob()
                    download_data.readinto(download_file)
                
                print(f"Downloaded {blob.name} to: {temp_file_path}")
                
                # Process the document
                self.process_document(temp_file_path)
                print(f"Successfully processed {blob.name}")
                processed_count += 1
                    
            except Exception as e:
                error_msg = f"Error processing blob {blob.name}: {str(e)}"
                print(error_msg)
                errors_log.append({"blob": blob.name, "error": str(e)})
                error_count += 1
            
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        print(f"Warning: Could not clean up {temp_file_path}: {e}")
        
        print(f"\nCompleted: {processed_count} processed, {error_count} errors")
        
        # Log detailed error information if there were errors
        if errors_log:
            print(f"\nDetailed error log:")
            for i, error_info in enumerate(errors_log, 1):
                print(f"  {i}. Blob: {error_info['blob']}")
                print(f"     Error: {error_info['error']}")
        
        return {
            "processed_count": processed_count,
            "error_count": error_count,
            "errors": errors_log
        }


def ingest_directory(adapter: DocumentIngestionOrchestrator, directory: str):
    """
    Classifies and ingests all files in the specified directory using the adapter.
    Prints results for each file.
    """
    dir_results = adapter.analyze_directory(directory)
    print("Directory classification results:")
    
    processed_count = error_count = 0
    errors_log = []
    
    for filename, info in dir_results.items():
        if not isinstance(info, dict) or "path" not in info:
            print(f"Skipping {filename}: {info}")
            continue
        file_path = info["path"]
        print(f"\nIngesting {filename} ({file_path})...")
        try:
            ingest_result = adapter.process_document(file_path)
            print(f"Ingestion successfully done for {filename}")
            processed_count += 1
        except Exception as e:
            error_msg = f"Error ingesting {filename}: {str(e)}"
            print(error_msg)
            errors_log.append({"file": filename, "path": file_path, "error": str(e)})
            error_count += 1
    
    print(f"\nDirectory ingestion completed: {processed_count} processed, {error_count} errors")
    
    # Log detailed error information if there were errors
    if errors_log:
        print(f"\nDetailed error log:")
        for i, error_info in enumerate(errors_log, 1):
            print(f"  {i}. File: {error_info['file']}")
            print(f"     Path: {error_info['path']}")
            print(f"     Error: {error_info['error']}")
    
    return {
        "processed_count": processed_count,
        "error_count": error_count,
        "errors": errors_log
    }


def ingest_azure_blob_container(adapter: DocumentIngestionOrchestrator, storage_account_name: str, storage_account_key: str, container_name: str = "rag-agents-container"):
    """Downloads and processes all files from Azure Blob Storage container using the adapter."""
    adapter.ingest_from_azure_blob(storage_account_name, storage_account_key, container_name)


# Example usage:
# if __name__ == "__main__":
#     adapter = DocumentIngestionOrchestrator()     
#     
#     # Local directory
#     ingest_directory(adapter, "docs")
#     
#     # Azure Blob Storage
#     ingest_azure_blob_container(adapter, "storage_account", "storage_key", "rag-agents-container")
