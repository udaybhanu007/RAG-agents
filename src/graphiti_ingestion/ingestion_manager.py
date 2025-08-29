import logging
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import mimetypes
import sys
import os
from datetime import datetime

from .config.ingestion_config import GraphitiIngestionConfig
from .services.blob_storage_service import AzureBlobStorageService, DocumentMetadata
from .services.graphiti_ingestion_service import GraphitiIngestionService
from .utils.document_chunker import DocumentChunker

# Import document container from the document_schema module
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.document_schema import DocumentContainer

logger = logging.getLogger(__name__)

class GraphitiIngestionManager:
    """
    Main class for managing Graphiti document ingestion from Azure Blob Storage
    
    This class orchestrates the entire ingestion pipeline:
    1. Fetches documents from Azure Blob Storage using separate method
    2. Performs dynamic schema evolution based on document patterns
    3. Ingests documents into Graphiti knowledge graph
    
    Phase 1 Implementation - Ingestion Only (No retrieval logic)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Graphiti Ingestion Manager with rate limiting environment variables
        
        Args:
            config_path: Path to configuration file (optional, uses .env.dev by default)
        """
        # Set environment variables for rate limiting following guidelines
        os.environ['SEMAPHORE_LIMIT'] = '1'  # Force sequential processing
        os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'  # Reduce API overhead
        
        # Load and validate configuration from .env.dev
        logger.info("Loading Graphiti Ingestion configuration...")
        self.config = GraphitiIngestionConfig()
        
        logger.info("Validating configuration and setting up Neo4j for Graphiti...")
        if not self.config.validate_all_configs():
            raise ValueError("Invalid configuration or Neo4j setup failed. Please check .env.dev file and Neo4j connection.")
        
        # Initialize services with proper separation of concerns
        logger.info("Initializing Azure Blob Storage service...")
        self.blob_storage_service = AzureBlobStorageService(self.config.azure_blob)
        
        logger.info("Initializing Graphiti ingestion service...")
        self.graphiti_service = GraphitiIngestionService(self.config)
        
        logger.info("Initializing document chunker with semantic chunking...")
        self.document_chunker = DocumentChunker(max_chunk_size=12000)  # Larger chunks for better semantic coherence
        
        logger.info("Graphiti Ingestion Manager initialized successfully")
        logger.info(f"Rate limiting: SEMAPHORE_LIMIT=1 (sequential processing)")
        logger.info(f"Using embedding model: {self.config.azure_openai.embedding_model}")
        logger.info(f"Semantic chunk size: 12KB for better rate limiting")
        logger.info(f"Neo4j database ready for Graphiti operations")
    
    
    
    async def check_blob_storage_connection(self) -> Dict[str, Any]:
        """
        Check Azure Blob Storage connection and return document count
        
        Returns:
            Dict containing connection status and document count
        """
        try:
            logger.info("Testing Azure Blob Storage connection...")
            
            # Check if container exists
            if not self.blob_storage_service.check_container_exists():
                raise Exception(f"Container '{self.config.azure_blob.container_name}' does not exist or is not accessible")
            
            logger.info(f"✅ Successfully connected to Azure Blob Storage container: {self.config.azure_blob.container_name}")
            
            # Get document list
            documents_metadata = self.blob_storage_service.list_all_documents()
            document_count = len(documents_metadata)
            
            logger.info(f"📊 Found {document_count} documents in Azure Blob Storage")
            
            # Log document details
            if documents_metadata:
                logger.info("📋 Document details:")
                for i, doc in enumerate(documents_metadata[:5], 1):  # Show first 5 documents
                    size_mb = doc.size / (1024 * 1024) if doc.size else 0
                    logger.info(f"  {i}. {doc.name} ({size_mb:.2f} MB, {doc.content_type})")
                
                if document_count > 5:
                    logger.info(f"  ... and {document_count - 5} more documents")
            
            return {
                "success": True,
                "document_count": document_count,
                "documents_metadata": documents_metadata,
                "container_name": self.config.azure_blob.container_name
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Azure Blob Storage: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_count": 0,
                "documents_metadata": []
            }

    async def ingest_large_document(self, document_metadata: DocumentMetadata, content: str) -> Dict[str, Any]:
        """
        Ingest large document with rate limiting following guidelines
        
        Args:
            document_metadata: Document metadata from blob storage
            content: Document content as string
            
        Returns:
            Ingestion result with comprehensive details
        """
        document_name = document_metadata.name
        logger.info("Starting ingestion of {document_name}")
        
        try:
            # Check if document needs chunking
            if len(content) < 50000:  # Under 50KB - single episode approach
                logger.info("Processing {document_name} as single episode (size: {len(content)} chars)")
                
                document_container = self._create_document_container_from_content(document_metadata, content)
                result = await self.graphiti_service.ingest_single_document(document_container)
                
                return {
                    "document_name": document_name,
                    "approach": "single_episode",
                    "chunks_processed": 1,
                    "success": result["success"],
                    "details": result
                }
            else:
                # Use semantic chunking approach for large documents
                logger.info("Processing {document_name} with semantic chunking (size: {len(content)} chars)")
                
                chunks = self.document_chunker.create_semantic_chunks(content, document_name)
                logger.info("Created {len(chunks)} semantic chunks")
                
                successful_chunks = 0
                failed_chunks = []
                
                for i, chunk in enumerate(chunks):
                    try:
                        logger.info("Processing chunk {i+1}/{len(chunks)}")
                        
                        # Create chunk metadata
                        chunk_metadata = document_metadata.to_dict()
                        chunk_metadata.update({
                            "title": f"{Path(document_name).stem} (Part {i+1}/{len(chunks)})",
                            "chunk_index": i+1,
                            "total_chunks": len(chunks)
                        })
                        
                        chunk_container = self._create_document_container_from_content_dict(chunk_metadata, chunk)
                        
                        result = await self.graphiti_service.ingest_single_document(chunk_container)
                        
                        if result["success"]:
                            successful_chunks += 1
                            logger.info("Successfully processed chunk {i+1}")
                        else:
                            failed_chunks.append((i+1, result.get("error", "Unknown error")))
                            logger.error("Failed chunk {i+1}: {result.get('error', 'Unknown error')}")
                        
                        # Add delay between chunks for rate limiting (following guidelines)
                        if i < len(chunks) - 1:  # Don't sleep after last chunk
                            logger.info("Waiting 5 seconds between chunks for rate limiting...")
                            await asyncio.sleep(5)
                            
                    except Exception as e:
                        logger.error(f"❌ Exception in chunk {i+1}: {str(e)}")
                        failed_chunks.append((i+1, str(e)))
                        
                        # For rate limit errors, wait longer
                        if "rate limit" in str(e).lower():
                            logger.info("⏸️ Extended wait due to rate limit...")
                            await asyncio.sleep(120)  # 2 minute wait
                
                return {
                    "document_name": document_name,
                    "approach": "semantic_chunking",
                    "total_chunks": len(chunks),
                    "successful_chunks": successful_chunks,
                    "failed_chunks": failed_chunks,
                    "success_rate": successful_chunks / len(chunks) * 100,
                    "success": successful_chunks > 0
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to ingest document {document_name}: {e}")
            return {
                "document_name": document_name,
                "success": False,
                "error": str(e)
            }
    
    async def ingest_all_documents_from_blob_storage(self, 
                                                   file_extensions: Optional[List[str]] = None,
                                                   prefix_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Simplified document ingestion following rate limiting guidelines
        
        Args:
            file_extensions: List of file extensions to process (e.g., ['.pdf', '.csv'])
            prefix_filter: Only process files with this prefix
            
        Returns:
            Comprehensive ingestion results
        """
        try:
            logger.info("🚀 Starting simplified document ingestion with rate limiting")
            
            # Get all documents from blob storage
            blob_check_result = await self.check_blob_storage_connection()
            if not blob_check_result["success"]:
                return {
                    "status": "failed",
                    "error": "Azure Blob Storage connection failed",
                    "details": blob_check_result["error"]
                }
            
            documents_metadata = blob_check_result["documents_metadata"]
            
            # Apply filters
            if file_extensions or prefix_filter:
                filtered_documents = []
                for doc in documents_metadata:
                    if file_extensions:
                        doc_ext = Path(doc.name).suffix.lower()
                        if doc_ext not in [ext.lower() for ext in file_extensions]:
                            continue
                    if prefix_filter and not doc.name.startswith(prefix_filter):
                        continue
                    filtered_documents.append(doc)
                documents_metadata = filtered_documents
            
            if not documents_metadata:
                logger.warning("⚠️ No documents found matching the criteria")
                return {
                    "status": "completed",
                    "message": "No documents found matching criteria",
                    "total_documents": 0,
                    "successful_ingestions": 0
                }
            
            logger.info(f"📋 Processing {len(documents_metadata)} documents sequentially")
            
            all_results = []
            successful_count = 0
            
            for i, doc_metadata in enumerate(documents_metadata):
                logger.info(f"📄 Processing document {i+1}/{len(documents_metadata)}: {doc_metadata.name}")
                
                try:
                    # Download document content
                    content_bytes = self.blob_storage_service.download_document(doc_metadata.name)
                    content = self._convert_content_to_text(content_bytes, doc_metadata.content_type)
                    
                    # Ingest document
                    result = await self.ingest_large_document(doc_metadata, content)
                    all_results.append(result)
                    
                    if result.get("success", False):
                        successful_count += 1
                        logger.info(f"✅ Successfully ingested {doc_metadata.name}")
                    else:
                        logger.error(f"❌ Failed to ingest {doc_metadata.name}")
                    
                    # Add delay between documents for rate limiting
                    if i < len(documents_metadata) - 1:
                        logger.info("⏸️ Waiting 10 seconds between documents for rate limiting...")
                        await asyncio.sleep(10)
                
                except Exception as e:
                    logger.error(f"❌ Exception processing {doc_metadata.name}: {e}")
                    all_results.append({
                        "document_name": doc_metadata.name,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "status": "completed",
                "total_documents": len(documents_metadata),
                "successful_ingestions": successful_count,
                "failed_ingestions": len(documents_metadata) - successful_count,
                "success_rate": (successful_count / len(documents_metadata) * 100) if documents_metadata else 0,
                "detailed_results": all_results
            }
            
        except Exception as e:
            logger.error(f"❌ Fatal error in ingestion process: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "total_documents": 0,
                "successful_ingestions": 0
            }
        """
        Ingest documents one by one from Azure Blob Storage with detailed step-by-step logging
        
        Args:
            file_extensions: List of file extensions to process (e.g., ['.pdf', '.csv'])
            prefix_filter: Only process files with this prefix
            
        Returns:
            Comprehensive ingestion results with step-by-step details
        """
        try:
            logger.info("🚀 Starting step-by-step document ingestion from Azure Blob Storage")
            logger.info("=" * 80)
            
            # Step 1: Check Azure Blob Storage connection
            logger.info("📡 Step 1: Checking Azure Blob Storage connection...")
            blob_check_result = await self.check_blob_storage_connection()
            
            if not blob_check_result["success"]:
                logger.error("❌ Azure Blob Storage connection failed - terminating ingestion")
                return {
                    "status": "failed",
                    "error": "Azure Blob Storage connection failed",
                    "details": blob_check_result["error"],
                    "total_documents_found": 0,
                    "total_documents_processed": 0,
                    "successful_ingestions": 0,
                    "failed_ingestions": 0
                }
            
            # Apply filters to documents
            documents_metadata = blob_check_result["documents_metadata"]
            if file_extensions or prefix_filter:
                logger.info(f"🔍 Applying filters - Extensions: {file_extensions}, Prefix: {prefix_filter}")
                filtered_documents = []
                for doc in documents_metadata:
                    # Check file extension filter
                    if file_extensions:
                        doc_ext = Path(doc.name).suffix.lower()
                        if doc_ext not in [ext.lower() for ext in file_extensions]:
                            continue
                    
                    # Check prefix filter
                    if prefix_filter and not doc.name.startswith(prefix_filter):
                        continue
                    
                    filtered_documents.append(doc)
                
                documents_metadata = filtered_documents
                logger.info(f"📋 After filtering: {len(documents_metadata)} documents to process")
            
            if not documents_metadata:
                logger.warning("⚠️ No documents found matching the criteria")
                return {
                    "status": "completed",
                    "message": "No documents found matching criteria",
                    "total_documents_found": blob_check_result["document_count"],
                    "total_documents_processed": 0,
                    "successful_ingestions": 0,
                    "failed_ingestions": 0,
                    "filters_applied": {
                        "file_extensions": file_extensions,
                        "prefix_filter": prefix_filter
                    }
                }
            
            # Prioritize documents by size (smaller first to reduce rate limiting issues)
            logger.info("🔄 Prioritizing documents by size (smaller files first)...")
            large_files = ["ARXIV_V5_CHESTXRAY.pdf", "Data_Entry_2017.csv"]  # Known large files
            
            # Separate small and large documents
            small_docs = [doc for doc in documents_metadata if doc.name not in large_files]
            large_docs = [doc for doc in documents_metadata if doc.name in large_files]
            
            # Process small documents first, then large ones
            documents_metadata = small_docs + large_docs
            
            logger.info(f"📋 Processing order:")
            for i, doc in enumerate(documents_metadata, 1):
                size_indicator = "🔴 LARGE" if doc.name in large_files else "🟢 SMALL"
                logger.info(f"   {i}. {doc.name} {size_indicator}")
            
            # Step 2: Process documents one by one
            logger.info(f"📥 Step 2: Starting sequential document processing ({len(documents_metadata)} documents)")
            logger.info("=" * 80)
            
            successful_ingestions = 0
            processing_details = []
            
            for i, doc_metadata in enumerate(documents_metadata, 1):
                logger.info(f"📄 Processing document {i}/{len(documents_metadata)}: {doc_metadata.name}")
                
                try:
                    # Download document content
                    logger.info(f"⬇️  Downloading document content...")
                    content_bytes = self.blob_storage_service.fetch_document_content(doc_metadata.name)
                    size_mb = len(content_bytes) / (1024 * 1024)
                    logger.info(f"✅ Downloaded {size_mb:.2f} MB successfully")
                    
                    # Convert to text
                    logger.info(f"🔄 Converting content to text...")
                    content_text = self._convert_content_to_text(content_bytes, doc_metadata.content_type)
                    
                    # Check content length and process with chunking if necessary
                    original_length = len(content_text)
                    if len(content_text) > self.config.max_content_length:
                        content_text = content_text[:self.config.max_content_length]
                        logger.warning(f"⚠️  Content truncated from {original_length} to {len(content_text)} characters")
                    else:
                        logger.info(f"✅ Content length: {len(content_text)} characters (within limits)")
                    
                    # Split document into chunks for processing
                    logger.info(f"� Splitting document into manageable chunks...")
                    chunks = self.document_chunker.chunk_text(content_text, doc_metadata.name)
                    logger.info(f"✅ Document split into {len(chunks)} chunks")
                    
                    # Process each chunk with intelligent rate limiting
                    chunk_success_count = 0
                    
                    # Add initial delay before starting chunk processing
                    logger.info(f"⏳ Initial delay of 30 seconds before starting chunk processing...")
                    await asyncio.sleep(30)
                    
                    for chunk_idx, chunk_data in enumerate(chunks, 1):
                        logger.info(f"   📄 Processing chunk {chunk_idx}/{len(chunks)} ({len(chunk_data['content'])} chars)")
                        
                        # Estimate tokens for this chunk (rough estimate: 1 token ≈ 0.75 characters)
                        estimated_tokens = int(len(chunk_data['content']) / 0.75) + 2000  # Add larger buffer for system prompts
                        
                        # Always wait before each chunk (conservative approach)
                        wait_time = max(self.api_tracker.get_wait_time(), 20.0)  # Minimum 20 seconds between chunks
                        logger.info(f"   ⏳ Rate limit protection: waiting {wait_time:.1f} seconds...")
                        await asyncio.sleep(wait_time)
                        
                        # Double-check rate limits
                        while not self.api_tracker.can_make_request(estimated_tokens):
                            additional_wait = self.api_tracker.get_wait_time()
                            logger.info(f"   ⏳ Additional rate limit protection: waiting {additional_wait:.1f} seconds...")
                            await asyncio.sleep(additional_wait)
                        
                        # Create chunk-specific metadata
                        chunk_metadata = self.document_chunker.create_chunk_metadata(
                            doc_metadata.__dict__, chunk_data
                        )
                        
                        # Create document container for this chunk
                        chunk_container = self._create_document_container_from_content_dict(
                            chunk_metadata, chunk_data['content']
                        )
                        
                        # Ingest chunk into Graphiti with exponential backoff
                        try:
                            chunk_ingestion_result = await self.backoff_handler.execute_with_backoff(
                                self._ingest_single_chunk_with_tracking,
                                chunk_container,
                                estimated_tokens
                            )
                        except Exception as e:
                            # Ingestion failed even after retries - terminate immediately
                            error_msg = str(e)
                            logger.error(f"❌ CRITICAL: Chunk {chunk_idx} of document '{doc_metadata.name}' failed: {error_msg}")
                            logger.error(f"🚨 TERMINATING entire ingestion process due to chunk failure")
                            
                            return {
                                "status": "failed",
                                "error": f"Chunk ingestion failed for document: {doc_metadata.name}",
                                "failure_details": f"Chunk {chunk_idx}/{len(chunks)}: {error_msg}",
                                "total_documents_found": blob_check_result["document_count"],
                                "total_documents_processed": i,
                                "successful_ingestions": successful_ingestions,
                                "failed_ingestions": 1,
                                "processing_details": processing_details,
                                "failed_document": doc_metadata.name,
                                "api_usage": self._get_api_usage_stats()
                            }
                        
                        if chunk_ingestion_result.get("success", False):
                            chunk_success_count += 1
                            tokens_used = chunk_ingestion_result.get("tokens_used", estimated_tokens)
                            logger.info(f"   ✅ Chunk {chunk_idx} ingested successfully")
                            logger.info(f"      Episode ID: {chunk_ingestion_result.get('episode_id', 'N/A')}")
                            logger.info(f"      Tokens used: {tokens_used}")
                            logger.info(f"      API Usage: {self._get_api_usage_summary()}")
                        else:
                            # This should not happen due to exponential backoff, but handle gracefully
                            error_msg = chunk_ingestion_result.get("error", "Unknown chunk ingestion error")
                            logger.error(f"❌ CRITICAL: Chunk {chunk_idx} failed unexpectedly: {error_msg}")
                            logger.error(f"🚨 TERMINATING entire ingestion process due to chunk failure")
                            
                            return {
                                "status": "failed",
                                "error": f"Chunk ingestion failed for document: {doc_metadata.name}",
                                "failure_details": f"Chunk {chunk_idx}/{len(chunks)}: {error_msg}",
                                "total_documents_found": blob_check_result["document_count"],
                                "total_documents_processed": i,
                                "successful_ingestions": successful_ingestions,
                                "failed_ingestions": 1,
                                "processing_details": processing_details,
                                "failed_document": doc_metadata.name,
                                "api_usage": self._get_api_usage_stats()
                            }
                    
                    # All chunks processed successfully
                    if chunk_success_count == len(chunks):
                        successful_ingestions += 1
                        logger.info(f"✅ Document '{doc_metadata.name}' fully ingested ({chunk_success_count} chunks)!")
                        logger.info(f"📊 Total API usage so far: {self._get_api_usage_summary()}")
                        
                        processing_details.append({
                            "document_name": doc_metadata.name,
                            "status": "success",
                            "chunks_processed": chunk_success_count,
                            "total_chunks": len(chunks),
                            "content_length": len(content_text),
                            "processing_order": i,
                            "api_usage": self._get_api_usage_stats()
                        })
                    else:
                        # Should not reach here due to immediate termination above
                        logger.error(f"❌ Unexpected state: partial chunk success for {doc_metadata.name}")
                        return {
                            "status": "failed",
                            "error": f"Partial chunk ingestion for document: {doc_metadata.name}",
                            "failure_details": f"Only {chunk_success_count}/{len(chunks)} chunks successful",
                            "total_documents_found": blob_check_result["document_count"],
                            "total_documents_processed": i,
                            "successful_ingestions": successful_ingestions,
                            "failed_ingestions": 1,
                            "processing_details": processing_details,
                            "failed_document": doc_metadata.name,
                            "api_usage": self._get_api_usage_stats()
                        }
                    
                    logger.info(f"✅ Completed processing document {i}/{len(documents_metadata)}")
                    
                    # Add delay between documents to respect rate limits (except for last document)
                    if i < len(documents_metadata):
                        wait_time = max(self.api_tracker.get_wait_time(), 20.0)  # Minimum 20 seconds between documents
                        logger.info(f"⏳ Waiting {wait_time:.1f} seconds before processing next document...")
                        await asyncio.sleep(wait_time)
                    
                    logger.info("-" * 60)
                    
                except Exception as e:
                    # Processing failed - terminate immediately as requested
                    logger.error(f"❌ CRITICAL: Failed to process document '{doc_metadata.name}': {e}")
                    logger.error(f"🚨 TERMINATING entire ingestion process due to processing failure")
                    
                    return {
                        "status": "failed", 
                        "error": f"Processing failed for document: {doc_metadata.name}",
                        "failure_details": str(e),
                        "total_documents_found": blob_check_result["document_count"],
                        "total_documents_processed": i,  # Include the failed document
                        "successful_ingestions": successful_ingestions,
                        "failed_ingestions": 1,
                        "processing_details": processing_details,
                        "failed_document": doc_metadata.name
                    }
            
            # All documents processed successfully
            logger.info("🎉 ALL DOCUMENTS PROCESSED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"📊 Final Results:")
            logger.info(f"   Total documents found in blob storage: {blob_check_result['document_count']}")
            logger.info(f"   Documents matching filters: {len(documents_metadata)}")
            logger.info(f"   Successfully processed: {successful_ingestions}")
            logger.info(f"   Success rate: 100%")
            
            return {
                "status": "completed",
                "total_documents_found": blob_check_result["document_count"],
                "total_documents_processed": len(documents_metadata),
                "successful_ingestions": successful_ingestions,
                "failed_ingestions": 0,
                "success_rate": 100.0,
                "processing_details": processing_details,
                "filters_applied": {
                    "file_extensions": file_extensions,
                    "prefix_filter": prefix_filter
                },
                "configuration_used": {
                    "batch_size": "N/A (sequential processing)",
                    "max_content_length": self.config.max_content_length,
                    "embedding_model": self.config.azure_openai.embedding_model
                }
            }
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Step-by-step ingestion process failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def ingest_all_documents_from_blob_storage(self, 
                                                   file_extensions: Optional[List[str]] = None,
                                                   prefix_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest all documents from Azure Blob Storage into Graphiti knowledge graph
        
        Args:
            file_extensions: List of file extensions to process (e.g., ['.pdf', '.csv'])
            prefix_filter: Only process files with this prefix
            
        Returns:
            Comprehensive ingestion results
        """
        try:
            logger.info("Starting complete document ingestion from Azure Blob Storage")
            
            # Step 1: Fetch documents from Azure Blob Storage using separate method
            logger.info("Fetching document list from Azure Blob Storage...")
            documents_metadata = self.blob_storage_service.list_all_documents(
                prefix=prefix_filter,
                file_extensions=file_extensions
            )
            
            if not documents_metadata:
                logger.warning("No documents found matching the criteria in Azure Blob Storage")
                return {
                    "status": "completed",
                    "total_documents": 0,
                    "message": "No documents found in Azure Blob Storage",
                    "filters_applied": {
                        "file_extensions": file_extensions,
                        "prefix_filter": prefix_filter
                    }
                }
            
            logger.info(f"Found {len(documents_metadata)} documents to process from Azure Blob Storage")
            
            # Step 2: Download and process documents
            logger.info("Downloading and processing documents from Azure Blob Storage...")
            document_schemas = []
            failed_processing = []
            
            for doc_metadata in documents_metadata:
                try:
                    # Fetch document content from Azure Blob Storage
                    content_bytes = self.blob_storage_service.fetch_document_content(doc_metadata.name)
                    
                    # Convert to text based on content type
                    content_text = self._convert_content_to_text(content_bytes, doc_metadata.content_type)
                    
                    # Respect max content length configuration
                    if len(content_text) > self.config.max_content_length:
                        content_text = content_text[:self.config.max_content_length]
                        logger.warning(f"Content truncated for {doc_metadata.name} (exceeded {self.config.max_content_length} chars)")
                    
                    # Create document container for Graphiti ingestion
                    doc_container = self._create_document_container_from_content(doc_metadata, content_text)
                    document_schemas.append(doc_container)
                    
                except Exception as e:
                    logger.error(f"Failed to process document {doc_metadata.name}: {e}")
                    failed_processing.append({
                        "document_name": doc_metadata.name,
                        "error": str(e),
                        "document_size": doc_metadata.size
                    })
            
            logger.info(f"Successfully processed {len(document_schemas)} documents for Graphiti ingestion")
            
            # Step 3: Ingest into Graphiti with sequential processing (rate limiting compliant)
            if document_schemas:
                logger.info("Starting Graphiti ingestion with sequential processing...")
                
                successful_ingestions = 0
                failed_ingestions = []
                
                for i, doc_schema in enumerate(document_schemas):
                    try:
                        logger.info(f"Processing document {i+1}/{len(document_schemas)}: {getattr(doc_schema, 'title', 'Unknown')}")
                        
                        result = await self.graphiti_service.ingest_single_document(doc_schema)
                        
                        if result["success"]:
                            successful_ingestions += 1
                            logger.info(f"Successfully ingested document {i+1}")
                        else:
                            failed_ingestions.append({
                                "document": getattr(doc_schema, 'title', 'Unknown'),
                                "error": result.get("error", "Unknown error")
                            })
                            logger.error(f"Failed to ingest document {i+1}: {result.get('error', 'Unknown error')}")
                        
                        # Add delay between documents for rate limiting
                        if i < len(document_schemas) - 1:
                            logger.info("Waiting 10 seconds between documents for rate limiting...")
                            await asyncio.sleep(10)
                            
                    except Exception as e:
                        logger.error(f"Exception ingesting document {i+1}: {str(e)}")
                        failed_ingestions.append({
                            "document": getattr(doc_schema, 'title', 'Unknown'),
                            "error": str(e)
                        })
                
                ingestion_results = {
                    "total_documents": len(document_schemas),
                    "successful_ingestions": successful_ingestions,
                    "failed_ingestions": len(failed_ingestions),
                    "failure_details": failed_ingestions,
                    "success_rate": (successful_ingestions / len(document_schemas) * 100) if document_schemas else 0
                }
            else:
                ingestion_results = {
                    "total_documents": 0,
                    "successful_ingestions": 0,
                    "failed_ingestions": len(failed_processing),
                    "failure_details": failed_processing
                }
            
            # Step 4: Compile comprehensive final results
            final_results = {
                "ingestion_summary": ingestion_results,
                "processing_failures": failed_processing,
                "total_documents_found_in_blob": len(documents_metadata),
                "total_documents_processed": len(document_schemas),
                "total_processing_failures": len(failed_processing),
                "filters_applied": {
                    "file_extensions": file_extensions,
                    "prefix_filter": prefix_filter
                },
                "configuration_used": {
                    "batch_size": self.config.batch_size,
                    "max_content_length": self.config.max_content_length,
                    "embedding_model": self.config.azure_openai.embedding_model
                },
                "status": "completed"
            }
            
            logger.info(f"Complete ingestion process finished successfully")
            logger.info(f"Documents found in blob storage: {final_results['total_documents_found_in_blob']}")
            logger.info(f"Documents successfully processed: {final_results['total_documents_processed']}")
            
            if ingestion_results.get("successful_ingestions"):
                logger.info(f"Documents successfully ingested to Graphiti: {ingestion_results['successful_ingestions']}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Complete ingestion process failed: {e}")
            raise
        
    def _create_document_container_from_content(self, document_metadata: DocumentMetadata, 
                                           content: str) -> DocumentContainer:
        """
        Create a simple document container for Graphiti ingestion
        
        Args:
            document_metadata: Document metadata from blob storage
            content: Document content as string
            
        Returns:
            DocumentContainer instance ready for Graphiti ingestion
        """
        # Extract basic information
        document_id = Path(document_metadata.name).stem
        title = document_id.replace('_', ' ').replace('-', ' ').title()
        
        # Create simple document container - Graphiti handles everything else dynamically
        return DocumentContainer(
            document_id=document_id,
            title=title,
            content=content,
            source_path=document_metadata.name,
            metadata=document_metadata.to_dict()
        )
    
    def _create_document_container_from_content_dict(self, metadata_dict: Dict[str, Any], 
                                                content: str) -> DocumentContainer:
        """
        Create a document container from a metadata dictionary (used for chunks)
        
        Args:
            metadata_dict: Document metadata as dictionary
            content: Document content as string
            
        Returns:
            DocumentContainer instance ready for Graphiti ingestion
        """
        # Extract basic information with chunk awareness
        document_id = Path(metadata_dict.get("name", "unknown")).stem
        title = metadata_dict.get("title", document_id.replace('_', ' ').replace('-', ' ').title())
        
        # Create document container
        return DocumentContainer(
            document_id=document_id,
            title=title,
            content=content,
            source_path=metadata_dict.get("name", "unknown"),
            metadata=metadata_dict
        )
    
    def _convert_content_to_text(self, content_bytes: bytes, content_type: str) -> str:
        """
        Convert document content bytes to text based on content type
        
        Args:
            content_bytes: Document content as bytes
            content_type: MIME type of the content
            
        Returns:
            Document content as text string
        """
        try:
            if content_type.startswith('text/') or 'csv' in content_type:
                return content_bytes.decode('utf-8')
            elif content_type == 'application/pdf':
                # For PDF files, return as text (basic implementation)
                # In a production environment, you might want to use PDF extraction libraries
                return content_bytes.decode('utf-8', errors='ignore')
            else:
                # For other file types, attempt UTF-8 decoding with error handling
                return content_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"Failed to convert content to text: {e}")
            return f"[Content conversion failed: {str(e)}]"
        except Exception as e:
            logger.warning(f"Failed to convert content to text: {e}")
            return f"[Content conversion failed: {str(e)}]"
    
    async def close(self):
        """Clean up all resources and close connections"""
        try:
            await self.graphiti_service.close_connections()
            logger.info("Graphiti Ingestion Manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing Graphiti Ingestion Manager: {e}")
