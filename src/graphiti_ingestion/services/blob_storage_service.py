import logging
from typing import List, Dict, Any, Optional, Iterator
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import AzureError
from io import BytesIO
import mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..config.ingestion_config import AzureBlobStorageConfig

logger = logging.getLogger(__name__)

class DocumentMetadata:
    """Document metadata from Azure blob storage"""
    
    def __init__(self, name: str, size: int, content_type: str, last_modified: str, etag: str):
        self.name = name
        self.size = size
        self.content_type = content_type
        self.last_modified = last_modified
        self.etag = etag
        self.file_extension = Path(name).suffix.lower()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "name": self.name,
            "size": self.size,
            "content_type": self.content_type,
            "last_modified": self.last_modified,
            "etag": self.etag,
            "file_extension": self.file_extension
        }

class AzureBlobStorageService:
    """
    Service for fetching documents from Azure Blob Storage
    
    This service handles all Azure Blob Storage operations for document ingestion
    without any retrieval logic as per guidelines.
    """
    
    def __init__(self, config: AzureBlobStorageConfig):
        """Initialize Azure Blob Storage Service with configuration"""
        self.config = config
        self._blob_service_client: Optional[BlobServiceClient] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize BlobServiceClient during initialization"""
        try:
            self._blob_service_client = BlobServiceClient.from_connection_string(
                self.config.get_connection_string()
            )
            
            # Test connection
            container_client = self._blob_service_client.get_container_client(
                self.config.container_name
            )
            if not container_client.exists():
                logger.warning(f"Container {self.config.container_name} does not exist")
            
            logger.info("Azure Blob Storage service initialized successfully")
            
        except AzureError as e:
            logger.error(f"Failed to initialize Azure Blob Storage service: {e}")
            raise
    
    def list_all_documents(self, prefix: Optional[str] = None, 
                          file_extensions: Optional[List[str]] = None) -> List[DocumentMetadata]:
        """
        List all documents in the Azure Blob Storage container
        
        Args:
            prefix: Filter blobs by name prefix
            file_extensions: List of file extensions to filter (e.g., ['.pdf', '.csv'])
        
        Returns:
            List of DocumentMetadata objects
        """
        try:
            container_client = self._blob_service_client.get_container_client(
                self.config.container_name
            )
            
            blobs = container_client.list_blobs(name_starts_with=prefix)
            documents = []
            
            for blob in blobs:
                doc_metadata = DocumentMetadata(
                    name=blob.name,
                    size=blob.size,
                    content_type=blob.content_settings.content_type or 
                                mimetypes.guess_type(blob.name)[0] or 'application/octet-stream',
                    last_modified=blob.last_modified.isoformat(),
                    etag=blob.etag
                )
                
                # Filter by file extensions if specified
                if file_extensions and doc_metadata.file_extension not in file_extensions:
                    continue
                
                documents.append(doc_metadata)
            
            logger.info(f"Found {len(documents)} documents in container {self.config.container_name}")
            return documents
            
        except AzureError as e:
            logger.error(f"Failed to list documents: {e}")
            raise
    
    def fetch_document_content(self, blob_name: str) -> bytes:
        """
        Fetch document content from Azure Blob Storage
        
        Args:
            blob_name: Name of the blob to download
            
        Returns:
            Document content as bytes
        """
        try:
            blob_client = self._blob_service_client.get_blob_client(
                container=self.config.container_name,
                blob=blob_name
            )
            
            download_stream = blob_client.download_blob()
            content = download_stream.readall()
            
            logger.info(f"Downloaded document: {blob_name} ({len(content)} bytes)")
            return content
            
        except AzureError as e:
            logger.error(f"Failed to download document {blob_name}: {e}")
            raise
    
    def fetch_document_content_stream(self, blob_name: str) -> Iterator[bytes]:
        """
        Fetch document as a stream for large files
        
        Args:
            blob_name: Name of the blob to download
            
        Yields:
            Chunks of document content
        """
        try:
            blob_client = self._blob_service_client.get_blob_client(
                container=self.config.container_name,
                blob=blob_name
            )
            
            download_stream = blob_client.download_blob()
            
            for chunk in download_stream.chunks():
                yield chunk
                
            logger.info(f"Streamed document: {blob_name}")
            
        except AzureError as e:
            logger.error(f"Failed to stream document {blob_name}: {e}")
            raise
    
    def get_document_metadata(self, blob_name: str) -> DocumentMetadata:
        """
        Get metadata for a specific document
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            DocumentMetadata object
        """
        try:
            blob_client = self._blob_service_client.get_blob_client(
                container=self.config.container_name,
                blob=blob_name
            )
            
            properties = blob_client.get_blob_properties()
            
            return DocumentMetadata(
                name=blob_name,
                size=properties.size,
                content_type=properties.content_settings.content_type or 
                            mimetypes.guess_type(blob_name)[0] or 'application/octet-stream',
                last_modified=properties.last_modified.isoformat(),
                etag=properties.etag
            )
            
        except AzureError as e:
            logger.error(f"Failed to get metadata for document {blob_name}: {e}")
            raise
    
    def batch_fetch_documents(self, blob_names: List[str], 
                            max_workers: int = 5) -> Dict[str, bytes]:
        """
        Fetch multiple documents in parallel from Azure Blob Storage
        
        Args:
            blob_names: List of blob names to download
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            Dictionary mapping blob names to their content
        """
        results = {}
        failed_downloads = []
        
        def fetch_single_document(blob_name: str) -> tuple:
            try:
                content = self.fetch_document_content(blob_name)
                return blob_name, content, None
            except Exception as e:
                return blob_name, None, str(e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_blob = {
                executor.submit(fetch_single_document, blob_name): blob_name 
                for blob_name in blob_names
            }
            
            for future in as_completed(future_to_blob):
                blob_name, content, error = future.result()
                
                if error:
                    failed_downloads.append((blob_name, error))
                    logger.error(f"Failed to download {blob_name}: {error}")
                else:
                    results[blob_name] = content
        
        if failed_downloads:
            logger.warning(f"Failed to download {len(failed_downloads)} documents")
        
        logger.info(f"Successfully downloaded {len(results)} documents")
        return results
    
    def check_container_exists(self) -> bool:
        """Check if the configured container exists"""
        try:
            container_client = self._blob_service_client.get_container_client(
                self.config.container_name
            )
            return container_client.exists()
        except AzureError as e:
            logger.error(f"Failed to check container existence: {e}")
            return False
    
    def get_container_info(self) -> Dict[str, Any]:
        """Get information about the container"""
        try:
            container_client = self._blob_service_client.get_container_client(
                self.config.container_name
            )
            
            properties = container_client.get_container_properties()
            
            return {
                "container_name": self.config.container_name,
                "last_modified": properties.last_modified.isoformat(),
                "etag": properties.etag,
                "exists": True
            }
            
        except AzureError as e:
            logger.error(f"Failed to get container info: {e}")
            return {
                "container_name": self.config.container_name,
                "exists": False,
                "error": str(e)
            }
