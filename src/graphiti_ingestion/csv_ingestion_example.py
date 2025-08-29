"""
CSV-Specific Ingestion Example for Large CSV Files
Demonstrates batch processing approach for large CSV datasets
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graphiti_ingestion.csv_ingestion_manager import CSVGraphitiIngestionManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """
    Main function demonstrating CSV batch ingestion
    """
    try:
        logger.info("=== CSV Batch Ingestion Example ===")
        logger.info("Processing large CSV files in manageable batches")
        
        # Set environment variables for rate limiting
        os.environ["SEMAPHORE_LIMIT"] = "1"
        os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
        
        logger.info("Environment: SEMAPHORE_LIMIT=1 (sequential processing)")
        logger.info("Environment: GRAPHITI_TELEMETRY_ENABLED=false")
        
        # Initialize CSV-specific ingestion manager with batch size
        logger.info("Initializing CSV Ingestion Manager with batch processing...")
        csv_ingestion_manager = CSVGraphitiIngestionManager(batch_size=500)  # 500 records per batch
        
        # Test Azure Blob Storage connection
        logger.info("\n=== Testing Azure Blob Storage Connection ===")
        connection_result = await csv_ingestion_manager.check_blob_storage_connection()
        
        if not connection_result["success"]:
            logger.error(f"Failed to connect to Azure Blob Storage: {connection_result['error']}")
            return
        
        logger.info(f"Found {connection_result['document_count']} documents total")
        csv_docs = [doc for doc in connection_result["documents_metadata"] if doc.name.lower().endswith('.csv')]
        logger.info(f"Found {len(csv_docs)} CSV documents:")
        for doc in csv_docs:
            size_mb = doc.size / (1024 * 1024) if doc.size else 0
            logger.info(f"  - {doc.name} ({size_mb:.2f} MB)")
        
        # Process CSV files with batch ingestion
        logger.info("\n=== Starting CSV Batch Ingestion ===")
        logger.info("This will process CSV files in small batches to avoid rate limits")
        
        csv_results = await csv_ingestion_manager.ingest_csv_documents_from_blob_storage()
        
        # Display comprehensive results
        logger.info("\n=== CSV Batch Ingestion Results ===")
        logger.info(f"Status: {csv_results.get('status', 'unknown')}")
        logger.info(f"Total CSV documents: {csv_results.get('total_csv_documents', 0)}")
        logger.info(f"Successfully processed: {csv_results.get('successful_documents', 0)}")
        logger.info(f"Failed documents: {csv_results.get('failed_documents', 0)}")
        logger.info(f"Overall success rate: {csv_results.get('success_rate', 0):.1f}%")
        logger.info(f"Batch size used: {csv_results.get('batch_size_used', 'unknown')}")
        
        # Show detailed results for each CSV file
        detailed_results = csv_results.get('detailed_results', [])
        for result in detailed_results:
            logger.info(f"\n--- {result.get('document_name', 'Unknown')} ---")
            logger.info(f"Approach: {result.get('approach', 'unknown')}")
            logger.info(f"Success: {result.get('success', False)}")
            
            if result.get('success', False):
                logger.info(f"Total records: {result.get('total_records', 'unknown')}")
                logger.info(f"Total batches: {result.get('total_batches', 'unknown')}")
                logger.info(f"Successful batches: {result.get('successful_batches', 0)}")
                logger.info(f"Batch success rate: {result.get('success_rate', 0):.1f}%")
                
                # Show failed batches if any
                failed_batches = result.get('failed_batches', [])
                if failed_batches:
                    logger.warning(f"Failed batches ({len(failed_batches)}):")
                    for failed_batch in failed_batches[:3]:  # Show first 3 failures
                        logger.warning(f"  Batch {failed_batch.get('batch_number', '?')}: {failed_batch.get('error', 'Unknown error')}")
            else:
                logger.error(f"Error: {result.get('error', 'Unknown error')}")
        
        # Summary statistics
        if detailed_results:
            total_records_processed = sum(r.get('total_records', 0) for r in detailed_results if r.get('success', False))
            total_batches_processed = sum(r.get('total_batches', 0) for r in detailed_results if r.get('success', False))
            total_successful_batches = sum(r.get('successful_batches', 0) for r in detailed_results if r.get('success', False))
            
            logger.info(f"\n=== Summary Statistics ===")
            logger.info(f"Total CSV records processed: {total_records_processed:,}")
            logger.info(f"Total batches created: {total_batches_processed}")
            logger.info(f"Total successful batches: {total_successful_batches}")
            if total_batches_processed > 0:
                batch_success_rate = (total_successful_batches / total_batches_processed) * 100
                logger.info(f"Overall batch success rate: {batch_success_rate:.1f}%")
        
        logger.info("\n=== CSV Ingestion Complete ===")
        
    except Exception as e:
        logger.error(f"Fatal error in CSV ingestion example: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup resources
        try:
            await csv_ingestion_manager.close()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    asyncio.run(main())
