"""
Simplified Example Usage of Graphiti Ingestion Manager
Following Rate Limiting Guidelines

This example demonstrates the clean, rate-limiting-aware Graphiti ingestion 
following the guidelines for large PDF ingestion.

Key Features:
- Environment-based rate limiting (SEMAPHORE_LIMIT=1)
- Semantic chunking for large documents  
- Sequential processing with delays
- Comprehensive error handling
"""

import asyncio
import logging
import os
from pathlib import Path
import sys

# Add the project root to Python path to resolve imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.graphiti_ingestion.ingestion_manager import GraphitiIngestionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graphiti_ingestion.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """
    Main example demonstrating rate-limiting-aware Graphiti ingestion
    """
    ingestion_manager = None
    
    try:
        logger.info("=== Graphiti Rate-Limited Ingestion Example ===")
        logger.info("Following guidelines for large PDF ingestion with rate limiting")
        
        # Environment variables are set automatically in the manager
        logger.info(f"Environment: SEMAPHORE_LIMIT={os.getenv('SEMAPHORE_LIMIT', 'not set')}")
        logger.info(f"Environment: GRAPHITI_TELEMETRY_ENABLED={os.getenv('GRAPHITI_TELEMETRY_ENABLED', 'not set')}")
        
        # Initialize Graphiti Ingestion Manager (sets environment variables)
        logger.info("Initializing Graphiti Ingestion Manager with rate limiting...")
        ingestion_manager = GraphitiIngestionManager()
        
        # Test connection first
        logger.info("\n=== Testing Azure Blob Storage Connection ===")
        connection_result = await ingestion_manager.check_blob_storage_connection()
        
        if not connection_result["success"]:
            logger.error(f"Connection failed: {connection_result['error']}")
            return
        
        logger.info(f"✅ Found {connection_result['document_count']} documents")
        
        # Example 1: Ingest specific file types (PDF files for testing)
        logger.info("\n=== Example 1: Ingesting PDF Documents Only ===")
        pdf_results = await ingestion_manager.ingest_all_documents_from_blob_storage(
            file_extensions=['.pdf']
        )
        
        logger.info(f"📊 PDF Results:")
        logger.info(f"   Total documents: {pdf_results.get('total_documents', 0)}")
        logger.info(f"   Successful: {pdf_results.get('successful_ingestions', 0)}")
        logger.info(f"   Failed: {pdf_results.get('failed_ingestions', 0)}")
        logger.info(f"   Success rate: {pdf_results.get('success_rate', 0):.1f}%")
        
        # Example 2: Ingest CSV files
        logger.info("\n=== Example 2: Ingesting CSV Documents Only ===")
        csv_results = await ingestion_manager.ingest_all_documents_from_blob_storage(
            file_extensions=['.csv']
        )
        
        logger.info(f"📊 CSV Results:")
        logger.info(f"   Total documents: {csv_results.get('total_documents', 0)}")
        logger.info(f"   Successful: {csv_results.get('successful_ingestions', 0)}")
        logger.info(f"   Failed: {csv_results.get('failed_ingestions', 0)}")
        logger.info(f"   Success rate: {csv_results.get('success_rate', 0):.1f}%")
        
        # Example 3: Show detailed results for troubleshooting
        logger.info("\n=== Detailed Results Analysis ===")
        all_results = pdf_results.get('detailed_results', []) + csv_results.get('detailed_results', [])
        
        for result in all_results:
            document_name = result.get('document_name', 'Unknown')
            success = result.get('success', False)
            approach = result.get('approach', 'unknown')
            
            if success:
                if approach == 'single_episode':
                    logger.info(f"✅ {document_name}: Single episode")
                elif approach == 'semantic_chunking':
                    chunks = result.get('total_chunks', 0)
                    success_rate = result.get('success_rate', 0)
                    logger.info(f"✅ {document_name}: {chunks} chunks, {success_rate:.1f}% success")
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"❌ {document_name}: {error}")
        
        logger.info("\n=== Ingestion Complete ===")
        
    except Exception as e:
        logger.error(f"Fatal error in ingestion example: {e}")
        
    finally:
        # Clean up resources
        if ingestion_manager:
            try:
                await ingestion_manager.close()
                logger.info("Resources cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    asyncio.run(main())
