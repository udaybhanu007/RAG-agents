from .classify_document import classify_document, analyze_repository_documents
from .mixed_document import MixedDocumentIngestor
from .ingestion_structured_document import StructuredDocumentIngestor
from .ingestion_unstructured_document import UnstructuredDocumentIngestor
from .box_client import BoxClient
from .confluence_client import ConfluenceMCPClient
import os
import tempfile
import asyncio
from typing import TYPE_CHECKING, Optional

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

    def download_azure_blob(self, storage_account_name: str, storage_account_key: str, container_name: str = "rag-agents-container", download_dir: Optional[str] = None):
        """Downloads all files from Azure Blob Storage container to downloaded_content folder."""
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Blob Storage is not available. Please install azure-storage-blob: pip install azure-storage-blob")
        
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient(
            account_url=f"https://{storage_account_name}.blob.core.windows.net",
            credential=storage_account_key
        )
        
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create downloaded_content directory at root if not provided
        if download_dir is None:
            # Get the root directory (assuming we're in src/data_ingestion)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach project root
            download_dir = os.path.join(root_dir, "downloaded_content")
        
        os.makedirs(download_dir, exist_ok=True)
            
        print(f"Downloading files from Azure Blob Storage container: {container_name} to {download_dir}")
        
        downloaded_count = error_count = 0
        errors_log = []
        downloaded_files = []
        
        for blob in container_client.list_blobs():
            print(f"Downloading blob: {blob.name}")
            
            try:
                blob_filename = os.path.basename(blob.name)
                download_file_path = os.path.join(download_dir, blob_filename)
                
                # Download blob to downloaded_content folder
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
                with open(download_file_path, "wb") as download_file:
                    download_data = blob_client.download_blob()
                    download_data.readinto(download_file)
                
                print(f"Downloaded {blob.name} to: {download_file_path}")
                downloaded_files.append(download_file_path)
                downloaded_count += 1
                    
            except Exception as e:
                error_msg = f"Error downloading blob {blob.name}: {str(e)}"
                print(error_msg)
                errors_log.append({"blob": blob.name, "error": str(e)})
                error_count += 1
        
        print(f"\nDownload completed: {downloaded_count} downloaded, {error_count} errors")
        
        if errors_log:
            print(f"\nDetailed error log:")
            for i, error_info in enumerate(errors_log, 1):
                print(f"  {i}. Blob: {error_info['blob']}")
                print(f"     Error: {error_info['error']}")
        
        return {
            "download_dir": download_dir,
            "downloaded_files": downloaded_files,
            "downloaded_count": downloaded_count,
            "error_count": error_count,
            "errors": errors_log
        }

    def download_from_box(self, folder_name: str, download_dir: Optional[str] = None):
        """Downloads all files from Box folder to downloaded_content folder."""
        try:
            box_client = BoxClient()
            
            # Create downloaded_content directory at root if not provided
            if download_dir is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                root_dir = os.path.dirname(os.path.dirname(current_dir))
                download_dir = os.path.join(root_dir, "downloaded_content")
            
            os.makedirs(download_dir, exist_ok=True)
            
            print(f"Downloading files from Box folder: {folder_name} to {download_dir}")
            
            # Fetch documents from Box
            result = box_client.fetch_folder_documents(folder_name)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                return {
                    "download_dir": download_dir,
                    "downloaded_files": [],
                    "downloaded_count": 0,
                    "error_count": 1,
                    "errors": [{"source": "box", "error": result["error"]}]
                }
            
            downloaded_files = []
            error_count = 0
            errors_log = []
            
            # Move files from Box's temporary downloaded_content to our downloaded_content
            files_list = result.get("files", [])
            if not isinstance(files_list, list):
                files_list = []
                
            for file_info in files_list:
                if file_info.get("downloaded_file"):
                    try:
                        # Copy from Box's downloaded_content to our downloaded_content
                        source_path = file_info["downloaded_file"]
                        target_path = os.path.join(download_dir, file_info["name"])
                        
                        import shutil
                        shutil.copy2(source_path, target_path)
                        downloaded_files.append(target_path)
                        
                        # Clean up source file
                        os.remove(source_path)
                        print(f"Moved {file_info['name']} to downloaded_content")
                        
                    except Exception as e:
                        error_count += 1
                        errors_log.append({"file": file_info["name"], "error": str(e)})
                        print(f"Error moving {file_info['name']}: {e}")
            
            downloaded_count = len(downloaded_files)
            print(f"\nBox download completed: {downloaded_count} files, {error_count} errors")
            
            return {
                "download_dir": download_dir,
                "downloaded_files": downloaded_files,
                "downloaded_count": downloaded_count,
                "error_count": error_count,
                "errors": errors_log
            }
            
        except Exception as e:
            print(f"Error during Box ingestion: {e}")
            return {
                "download_dir": download_dir if 'download_dir' in locals() else "",
                "downloaded_files": [],
                "downloaded_count": 0,
                "error_count": 1,
                "errors": [{"source": "box_client", "error": str(e)}]
            }

    def download_from_confluence_pages(self, page_titles: list, download_dir: Optional[str] = None):
        """Downloads content from Confluence pages based on titles to downloaded_content folder."""
        try:
            # Import the ConfluenceMCPClient
            import sys
            import os as os_sys
            # Add Ingestion-POC to path
            ingestion_poc_path = os_sys.path.join(os_sys.path.dirname(os_sys.path.dirname(os_sys.path.dirname(__file__))), "Ingestion-POC")
            if ingestion_poc_path not in sys.path:
                sys.path.append(ingestion_poc_path)            
         
            
            # Create downloaded_content directory at root if not provided
            if download_dir is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                root_dir = os.path.dirname(os.path.dirname(current_dir))
                download_dir = os.path.join(root_dir, "downloaded_content")
            
            os.makedirs(download_dir, exist_ok=True)
            
            print(f"Downloading content from {len(page_titles)} Confluence pages to {download_dir}")
            
            # Run async function in sync context
            async def download_pages():
                client = ConfluenceMCPClient()
                return await client.download_multiple_pages(page_titles, download_dir)
            
            # Execute the async function
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(download_pages())
            
            # Convert to our expected format
            downloaded_files = result.get("downloaded_files", [])
            downloaded_count = result.get("successful_downloads", 0)
            error_count = result.get("failed_downloads", 0)
            
            errors_log = []
            for individual_result in result.get("individual_results", []):
                if not individual_result.get("success"):
                    errors_log.append({
                        "page": individual_result.get("page_title", "unknown"),
                        "error": individual_result.get("error", "unknown error")
                    })
            
            print(f"\nConfluence download completed: {downloaded_count} pages, {error_count} errors")
            
            return {
                "download_dir": download_dir,
                "downloaded_files": downloaded_files,
                "downloaded_count": downloaded_count,
                "error_count": error_count,
                "errors": errors_log
            }
            
        except Exception as e:
            print(f"Error during Confluence ingestion: {e}")
            return {
                "download_dir": download_dir if 'download_dir' in locals() else "",
                "downloaded_files": [],
                "downloaded_count": 0,
                "error_count": 1,
                "errors": [{"source": "confluence_client", "error": str(e)}]
            }

    def process_downloaded_folder(self, download_dir: str, cleanup: bool = False):
        """Processes all files from the downloaded_content folder using process_document method."""
        if not os.path.exists(download_dir):
            raise ValueError(f"Download directory does not exist: {download_dir}")
        
        print(f"Processing files from downloaded folder: {download_dir}")
        
        processed_count = error_count = 0
        errors_log = []
        
        for filename in os.listdir(download_dir):
            file_path = os.path.join(download_dir, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            print(f"Processing file: {filename}")
            
            try:
                self.process_document(file_path)
                print(f"Successfully processed {filename}")
                processed_count += 1
                
            except Exception as e:
                error_msg = f"Error processing file {filename}: {str(e)}"
                print(error_msg)
                errors_log.append({"file": filename, "path": file_path, "error": str(e)})
                error_count += 1
        
        # Cleanup downloaded files if requested
        if cleanup:
            try:
                import shutil
                shutil.rmtree(download_dir)
                print(f"Cleaned up download directory: {download_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up download directory {download_dir}: {e}")
        
        print(f"\nProcessing completed: {processed_count} processed, {error_count} errors")
        
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

    # def process_default_downloaded_folder(self, cleanup: bool = False):
    #     """Processes all files from the default downloaded_content folder at project root."""
    #     # Get the root directory (assuming we're in src/data_ingestion)
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     root_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach project root
    #     download_dir = os.path.join(root_dir, "downloaded_content")
        
    #     return self.process_downloaded_folder(download_dir, cleanup)


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


# Example usage:
# if __name__ == "__main__":
#     adapter = DocumentIngestionOrchestrator()     
#     
#     # Local directory
#     ingest_directory(adapter, "docs")
#     
#     # Azure Blob Storage - downloads to downloaded_content folder only
#     download_result = adapter.download_azure_blob("storage_account", "storage_key", "rag-agents-container")
#     
#     # Box folder - downloads to downloaded_content folder only
#     box_result = adapter.download_from_box("documents-ingest")
#     
#     # Confluence pages - downloads to downloaded_content folder only
#     confluence_result = adapter.download_from_confluence_pages(["MPC-POC", "Getting Started"])
#     
#     # Process files from downloaded_content folder separately
#     download_dir = download_result["download_dir"]
#     adapter.process_downloaded_folder(download_dir, cleanup=False)
