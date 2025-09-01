"""
CSV-Specific Ingestion Manager for Large CSV Files
Handles CSV files by processing them in batches rather than as single documents
"""

import asyncio
import logging
import pandas as pd
from io import StringIO
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

from .ingestion_manager import GraphitiIngestionManager
from ..document_schema.base_schema import DocumentContainer

logger = logging.getLogger(__name__)

class CSVGraphitiIngestionManager(GraphitiIngestionManager):
    """
    Extended GraphitiIngestionManager specifically for CSV file processing
    Processes CSV files in batches to avoid rate limiting issues
    """
    
    def __init__(self, batch_size: int = 1000):
        """
        Initialize CSV-specific ingestion manager
        
        Args:
            batch_size: Number of CSV records to process per batch (default: 1000)
        """
        super().__init__()
        self.batch_size = batch_size
        logger.info(f"CSV Ingestion Manager initialized with batch size: {batch_size}")
    
    async def ingest_csv_file_in_batches(self, 
                                       document_metadata,
                                       csv_content: str) -> Dict[str, Any]:
        """
        Process CSV file in batches for better rate limiting compliance
        
        Args:
            document_metadata: Document metadata from blob storage
            csv_content: CSV content as string
            
        Returns:
            Detailed ingestion results
        """
        document_name = document_metadata.name
        logger.info(f"Starting batch CSV ingestion for {document_name}")
        
        try:
            # Parse CSV content
            csv_data = pd.read_csv(StringIO(csv_content))
            total_records = len(csv_data)
            logger.info(f"CSV contains {total_records} records")
            
            # Calculate number of batches
            num_batches = (total_records + self.batch_size - 1) // self.batch_size
            logger.info(f"Will process in {num_batches} batches of {self.batch_size} records each")
            
            successful_batches = 0
            failed_batches = []
            batch_results = []
            
            # Process each batch
            for batch_num in range(num_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min((batch_num + 1) * self.batch_size, total_records)
                
                logger.info(f"Processing batch {batch_num + 1}/{num_batches}: records {start_idx}-{end_idx}")
                
                try:
                    # Extract batch data
                    batch_data = csv_data.iloc[start_idx:end_idx]
                    
                    # Create batch document
                    batch_result = await self._process_csv_batch(
                        document_metadata, 
                        batch_data, 
                        batch_num + 1, 
                        num_batches,
                        start_idx,
                        end_idx
                    )
                    
                    batch_results.append(batch_result)
                    
                    if batch_result.get("success", False):
                        successful_batches += 1
                        logger.info(f"Successfully processed batch {batch_num + 1}")
                    else:
                        failed_batches.append({
                            "batch_number": batch_num + 1,
                            "records_range": f"{start_idx}-{end_idx}",
                            "error": batch_result.get("error", "Unknown error")
                        })
                        logger.error(f"Failed batch {batch_num + 1}: {batch_result.get('error', 'Unknown error')}")
                    
                    # No delay for ultra-fast processing
                    # if batch_num < num_batches - 1:
                    #     logger.info("Waiting 1 second between batches...")
                    #     await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Exception in batch {batch_num + 1}: {str(e)}")
                    failed_batches.append({
                        "batch_number": batch_num + 1,
                        "records_range": f"{start_idx}-{end_idx}",
                        "error": str(e)
                    })
            
            return {
                "document_name": document_name,
                "approach": "csv_batch_processing",
                "total_records": total_records,
                "total_batches": num_batches,
                "batch_size": self.batch_size,
                "successful_batches": successful_batches,
                "failed_batches": failed_batches,
                "success_rate": (successful_batches / num_batches * 100) if num_batches > 0 else 0,
                "detailed_batch_results": batch_results,
                "success": successful_batches > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to process CSV file {document_name}: {e}")
            return {
                "document_name": document_name,
                "success": False,
                "error": str(e),
                "approach": "csv_batch_processing"
            }
    
    async def _process_csv_batch(self, 
                                document_metadata,
                                batch_data: pd.DataFrame,
                                batch_number: int,
                                total_batches: int,
                                start_idx: int,
                                end_idx: int) -> Dict[str, Any]:
        """
        Process a single batch of CSV records using bulk ingestion
        
        Args:
            document_metadata: Original document metadata
            batch_data: DataFrame containing batch records
            batch_number: Current batch number (1-indexed)
            total_batches: Total number of batches
            start_idx: Starting record index
            end_idx: Ending record index
            
        Returns:
            Batch processing result
        """
        try:
            # Create batch title for episode naming
            batch_title = f"{Path(document_metadata.name).stem} - Batch {batch_number}/{total_batches}"
            
            # Use the new bulk CSV ingestion method
            result = await self.graphiti_service.ingest_csv_bulk(
                csv_data=batch_data,
                document_metadata=document_metadata
            )
            
            return {
                "batch_number": batch_number,
                "records_range": f"{start_idx}-{end_idx}",
                "records_count": len(batch_data),
                "success": result.get("success", False),
                "graphiti_result": result,
                "title": batch_title,
                "episodes_created": result.get("episodes_ingested", 0)
            }
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_number}: {str(e)}")
            return {
                "batch_number": batch_number,
                "records_range": f"{start_idx}-{end_idx}",
                "success": False,
                "error": str(e)
            }
    
    def _format_csv_batch_content(self, 
                                 batch_data: pd.DataFrame, 
                                 batch_title: str,
                                 start_idx: int,
                                 end_idx: int) -> str:
        """
        Format CSV batch data into meaningful text content for knowledge graph ingestion
        
        Args:
            batch_data: DataFrame containing batch records
            batch_title: Title for this batch
            start_idx: Starting record index
            end_idx: Ending record index
            
        Returns:
            Formatted text content
        """
        try:
            content_lines = [
                f"=== {batch_title} ===",
                f"Records {start_idx} to {end_idx} from CSV dataset",
                f"Total records in batch: {len(batch_data)}",
                "",
                "=== Data Summary ===",
                f"Columns: {', '.join(batch_data.columns.tolist())}",
                ""
            ]
            
            # Add sample records (first 5 and last 5 if batch is large)
            if len(batch_data) <= 10:
                content_lines.append("=== All Records ===")
                for idx, (_, row) in enumerate(batch_data.iterrows()):
                    content_lines.append(f"Record {start_idx + idx + 1}:")
                    for col, value in row.items():
                        content_lines.append(f"  {col}: {value}")
                    content_lines.append("")
            else:
                # Show first 5 records
                content_lines.append("=== First 5 Records ===")
                for idx, (_, row) in enumerate(batch_data.head(5).iterrows()):
                    content_lines.append(f"Record {start_idx + idx + 1}:")
                    for col, value in row.items():
                        content_lines.append(f"  {col}: {value}")
                    content_lines.append("")
                
                # Show last 5 records
                content_lines.append("=== Last 5 Records ===")
                tail_start_idx = len(batch_data) - 5
                for idx, (_, row) in enumerate(batch_data.tail(5).iterrows()):
                    content_lines.append(f"Record {start_idx + tail_start_idx + idx + 1}:")
                    for col, value in row.items():
                        content_lines.append(f"  {col}: {value}")
                    content_lines.append("")
            
            # Add statistical summary if numerical columns exist
            numeric_cols = batch_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                content_lines.append("=== Statistical Summary ===")
                for col in numeric_cols:
                    stats = batch_data[col].describe()
                    content_lines.append(f"{col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']}, max={stats['max']}")
                content_lines.append("")
            
            # Add categorical summary
            categorical_cols = batch_data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                content_lines.append("=== Categorical Summary ===")
                for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                    value_counts = batch_data[col].value_counts().head(5)
                    content_lines.append(f"{col} top values: {dict(value_counts)}")
                content_lines.append("")
            
            return "\n".join(content_lines)
            
        except Exception as e:
            logger.error(f"Error formatting CSV batch content: {e}")
            return f"CSV Batch {batch_title}\nRecords {start_idx}-{end_idx}\nError formatting content: {str(e)}"
    
    async def ingest_csv_documents_from_blob_storage(self, 
                                                   batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Ingest all CSV documents from blob storage using batch processing
        
        Args:
            batch_size: Override default batch size for this ingestion
            
        Returns:
            Comprehensive ingestion results
        """
        if batch_size:
            self.batch_size = batch_size
            
        try:
            logger.info(f"Starting CSV batch ingestion from Azure Blob Storage (batch size: {self.batch_size})")
            
            # Get CSV documents from blob storage
            blob_check_result = await self.check_blob_storage_connection()
            if not blob_check_result["success"]:
                return {
                    "status": "failed",
                    "error": "Azure Blob Storage connection failed",
                    "details": blob_check_result["error"]
                }
            
            # Filter for CSV files only
            csv_documents = [doc for doc in blob_check_result["documents_metadata"] 
                           if doc.name.lower().endswith('.csv')]
            
            if not csv_documents:
                logger.warning("No CSV documents found in blob storage")
                return {
                    "status": "completed",
                    "message": "No CSV documents found",
                    "total_documents": 0,
                    "successful_ingestions": 0
                }
            
            logger.info(f"Found {len(csv_documents)} CSV documents to process")
            
            all_results = []
            successful_count = 0
            
            for i, doc_metadata in enumerate(csv_documents):
                logger.info(f"Processing CSV document {i+1}/{len(csv_documents)}: {doc_metadata.name}")
                
                try:
                    # Download CSV content
                    content_bytes = self.blob_storage_service.fetch_document_content(doc_metadata.name)
                    csv_content = content_bytes.decode('utf-8')
                    
                    # Process CSV in batches
                    result = await self.ingest_csv_file_in_batches(doc_metadata, csv_content)
                    all_results.append(result)
                    
                    if result.get("success", False):
                        successful_count += 1
                        logger.info(f"Successfully processed CSV: {doc_metadata.name}")
                    else:
                        logger.error(f"Failed to process CSV: {doc_metadata.name}")
                    
                    # Reduced delay between CSV files for faster processing
                    if i < len(csv_documents) - 1:
                        logger.info("Waiting 2 seconds between CSV files...")
                        await asyncio.sleep(2)
                
                except Exception as e:
                    logger.error(f"Exception processing CSV {doc_metadata.name}: {e}")
                    all_results.append({
                        "document_name": doc_metadata.name,
                        "success": False,
                        "error": str(e),
                        "approach": "csv_batch_processing"
                    })
            
            return {
                "status": "completed",
                "total_csv_documents": len(csv_documents),
                "successful_documents": successful_count,
                "failed_documents": len(csv_documents) - successful_count,
                "success_rate": (successful_count / len(csv_documents) * 100) if csv_documents else 0,
                "batch_size_used": self.batch_size,
                "detailed_results": all_results
            }
            
        except Exception as e:
            logger.error(f"Fatal error in CSV batch ingestion: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "total_documents": 0,
                "successful_ingestions": 0
            }
