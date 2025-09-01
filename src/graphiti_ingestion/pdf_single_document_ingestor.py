#!/usr/bin/env python3
"""
Single PDF Document Ingestion for Graphiti
==========================================

This script inges                  extraction_time = time.time() - start_time
            self.stats['extraction_time'] = extraction_time
            
            logger.info(f"✅ Content extracted:")
            logger.info(f"   Word count: {extracted_data['word_count']:,}")
            logger.info(f"   Pages: {extracted_data['pages_count']}")
            logger.info(f"   Extraction time: {extraction_time:.2f}s")     logger.info(f"✅ Content extracted:")
            logger.info(f"   Word count: {extracted_data['word_count']:,}")
            logger.info(f"   Pages: {extracted_data['pages_count']}")
            logger.info(f"   Extraction time: {extraction_time:.2f}s")     logger.info(f"✅ Content extracted:")
            logger.info(f"   Word count: {extracted_data['word_count']:,}")
            logger.info(f"   Pages: {extracted_data['pages_count']}")
            logger.info(f"   Extraction time: {extraction_time:.2f}s")     logger.info(f"[SUCCESS] Content extracted:")
            logger.info(f"   [STATS] Word count: {extracted_data['word_count']:,}")
            logger.info(f"   [PAGES] Pages: {extracted_data['pages_count']}")
            logger.info(f"   [TIME] Extraction time: {extraction_time:.2f}s") specific PDF file (README_CHESTXRAY.pdf) from the 
downloaded_content folder into Neo4j using Graphiti framework.

Key Features:
- Direct local file processing (no Azure Blob dependency)
- Rate limiting compliant (SEMAPHORE_LIMIT=1)
- Semantic chunking for large documents
- Comprehensive error handling and progress tracking
- Based on production bulk processing patterns
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import time

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.graphiti_ingestion.services.graphiti_ingestion_service import GraphitiIngestionService
    from src.graphiti_ingestion.config.ingestion_config import GraphitiIngestionConfig
    from graphiti_core.utils.bulk_utils import RawEpisode
    from graphiti_core.nodes import EpisodeType
    from dotenv import load_dotenv
    import PyPDF2  # For PDF extraction
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Make sure you have the required dependencies installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SinglePDFGraphitiProcessor:
    """
    Processes a single PDF file using Graphiti framework with semantic chunking.
    """
    
    def __init__(self):
        self.service = None
        self.stats = {
            'file_processed': False,
            'extraction_time': 0.0,
            'ingestion_time': 0.0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'total_characters': 0
        }
    
    async def initialize_services(self):
        """Initialize Graphiti and PDF extraction services."""
        try:
            logger.info("=== Initializing PDF Ingestion Services ===")
            
            # Load environment variables
            env_path = os.path.join(project_root, '.env.dev')
            if not os.path.exists(env_path):
                raise FileNotFoundError(f"Environment file not found: {env_path}")
            
            load_dotenv(dotenv_path=env_path, override=True)
            
            # Validate required environment variables
            required_vars = ['NEO4J_URI', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT']
            for var in required_vars:
                if not os.getenv(var):
                    raise ValueError(f"{var} environment variable is not set")
            
            # Set rate limiting environment variables
            os.environ['SEMAPHORE_LIMIT'] = '1'
            os.environ['GRAPHITI_TELEMETRY_ENABLED'] = 'false'
            
            # Initialize Graphiti service
            logger.info("Initializing Graphiti service...")
            config = GraphitiIngestionConfig()
            self.service = GraphitiIngestionService(config)
            
            logger.info("[SUCCESS] Services initialized successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize services: {e}")
            raise
    
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """Extract content from PDF file using PyPDF2."""
        try:
            logger.info(f"[PDF] Extracting content from: {Path(pdf_path).name}")
            start_time = time.time()
            
            extracted_data = {
                'text': '',
                'word_count': 0,
                'pages_count': 0,
                'table_count': 0  # Simplified - no table extraction for now
            }
            
            full_text = ""
            
            # Open PDF with PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                extracted_data['pages_count'] = len(pdf_reader.pages)
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
            
            extracted_data['text'] = full_text
            extracted_data['word_count'] = len(full_text.split()) if full_text else 0
            
            extraction_time = time.time() - start_time
            self.stats['extraction_time'] = extraction_time
            
            logger.info(f"✅ Content extracted:")
            logger.info(f"   📊 Word count: {extracted_data['word_count']:,}")
            logger.info(f"   � Pages: {extracted_data['pages_count']}")
            logger.info(f"   ⏱️ Extraction time: {extraction_time:.2f}s")
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"[ERROR] PDF extraction failed: {e}")
            raise
    
    def create_semantic_chunks(self, content: str, max_chunk_size: int = 8000) -> list:
        """
        Create semantic chunks from PDF content.
        Based on the guidelines for large PDF ingestion.
        """
        try:
            logger.info("[PROCESS] Creating semantic chunks...")
            
            # Split by major sections first
            section_markers = [
                "Background & Motivation:",
                "Details:",
                "Contents:",
                "Limitations:",
                "Acknowledgement:",
                "Reference:",
                "co-occurrence matrix",
                "## ",  # Markdown headers
                "# ",   # Main headers
                "\n\n"  # Paragraph breaks
            ]
            
            chunks = []
            current_chunk = ""
            current_size = 0
            
            lines = content.split('\n')
            for line in lines:
                # Check if this line starts a new section
                is_section_start = any(marker in line for marker in section_markers)
                
                if is_section_start and current_chunk and current_size > 2000:
                    # Save current chunk and start new one
                    chunks.append(current_chunk.strip())
                    current_chunk = line + '\n'
                    current_size = len(line)
                else:
                    # Add to current chunk if within size limit
                    if current_size + len(line) < max_chunk_size:
                        current_chunk += line + '\n'
                        current_size += len(line)
                    else:
                        # Chunk is full, save it and start new one
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = line + '\n'
                        current_size = len(line)
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Filter out very small chunks (less than 500 characters)
            meaningful_chunks = [chunk for chunk in chunks if len(chunk) >= 500]
            
            logger.info(f"[SUCCESS] Created {len(meaningful_chunks)} semantic chunks")
            logger.info(f"   [SIZE] Average chunk size: {sum(len(c) for c in meaningful_chunks) // len(meaningful_chunks):,} chars")
            
            return meaningful_chunks
            
        except Exception as e:
            logger.error(f"[ERROR] Chunking failed: {e}")
            raise
    
    async def ingest_pdf_as_episodes(self, pdf_path: str) -> Dict[str, Any]:
        """
        Ingest PDF file as multiple episodes using semantic chunking.
        """
        try:
            start_time = time.time()
            file_name = Path(pdf_path).name
            
            logger.info(f"Starting ingestion of: {file_name}")
            
            # Extract PDF content
            extracted_data = self.extract_pdf_content(pdf_path)
            content = extracted_data['text']
            
            if not content:
                raise ValueError("No text content extracted from PDF")
            
            self.stats['total_characters'] = len(content)
            
            # Create semantic chunks
            chunks = self.create_semantic_chunks(content)
            self.stats['total_chunks'] = len(chunks)
            
            if not chunks:
                raise ValueError("No meaningful chunks created from content")
            
            # Process chunks sequentially with rate limiting
            logger.info(f"Processing {len(chunks)} chunks sequentially...")
            successful_count = 0
            failed_episodes = []
            
            for i, chunk in enumerate(chunks, 1):
                try:
                    logger.info(f"[PROCESS] Processing chunk {i}/{len(chunks)} ({len(chunk):,} chars)")
                    
                    # Create episode for this chunk
                    episode = RawEpisode(
                        name=f"{Path(file_name).stem}_section_{i}",
                        content=f"Document: {file_name}\n"
                               f"Section {i} of {len(chunks)}\n\n"
                               f"{chunk}",
                        source_description=f"Section {i} of {file_name} - Medical research document",
                        source=EpisodeType.json,
                        reference_time=datetime.now(timezone.utc)
                    )
                    
                    # Add episode to Graphiti
                    await self.service._graphiti.add_episode(
                        name=episode.name,
                        episode_body=episode.content,
                        source_description=episode.source_description,
                        reference_time=episode.reference_time,
                        source=episode.source
                    )
                    
                    successful_count += 1
                    logger.info(f"[SUCCESS] Chunk {i} ingested successfully")
                    
                    # Rate limiting: wait between chunks
                    if i < len(chunks):
                        await asyncio.sleep(2.0)  # 2 second delay between chunks
                    
                except Exception as e:
                    logger.error(f"[ERROR] Failed to ingest chunk {i}: {e}")
                    failed_episodes.append({
                        'chunk_number': i,
                        'error': str(e)
                    })
            
            # Calculate final statistics
            total_time = time.time() - start_time
            self.stats['ingestion_time'] = total_time - self.stats['extraction_time']
            self.stats['successful_chunks'] = successful_count
            self.stats['file_processed'] = successful_count > 0
            
            success_rate = (successful_count / len(chunks)) * 100 if chunks else 0
            
            result = {
                'success': successful_count > 0,
                'file_name': file_name,
                'total_chunks': len(chunks),
                'successful_chunks': successful_count,
                'failed_chunks': len(failed_episodes),
                'success_rate': success_rate,
                'total_time': total_time,
                'extraction_time': self.stats['extraction_time'],
                'ingestion_time': self.stats['ingestion_time'],
                'total_characters': self.stats['total_characters'],
                'failed_episodes': failed_episodes
            }
            
            logger.info(f"[COMPLETE] Ingestion completed:")
            logger.info(f"   [FILE] File: {file_name}")
            logger.info(f"   [SUCCESS] Success rate: {success_rate:.1f}%")
            logger.info(f"   [CHUNKS] Chunks: {successful_count}/{len(chunks)}")
            logger.info(f"   [TIME] Total time: {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] PDF ingestion failed: {e}")
            raise
    
    def print_final_summary(self, result: Dict[str, Any]):
        """Print comprehensive ingestion summary."""
        logger.info("\n" + "="*60)
        logger.info("PDF INGESTION SUMMARY")
        logger.info("="*60)
        
        if result['success']:
            logger.info(f"[FILE] File: {result['file_name']}")
            logger.info(f"[STATUS] Status: SUCCESSFUL")
            logger.info(f"[STATS] Success Rate: {result['success_rate']:.1f}%")
            logger.info(f"[CHUNKS] Chunks Processed: {result['successful_chunks']}/{result['total_chunks']}")
            logger.info(f"[SIZE] Total Characters: {result['total_characters']:,}")
            logger.info(f"[TIME] Extraction Time: {result['extraction_time']:.2f}s")
            logger.info(f"[TIME] Ingestion Time: {result['ingestion_time']:.2f}s")
            logger.info(f"[TIME] Total Time: {result['total_time']:.2f}s")
            
            if result['failed_chunks'] > 0:
                logger.warning(f"[WARNING] Failed Chunks: {result['failed_chunks']}")
                
        else:
            logger.error("[STATUS] Status: FAILED")
            logger.error("[ERROR] No chunks were successfully ingested")
        
        logger.info("="*60)

async def main():
    """Main execution function."""
    processor = SinglePDFGraphitiProcessor()
    
    try:
        # Initialize services
        await processor.initialize_services()
        
        # Define the PDF file path
        pdf_path = os.path.join(project_root, "downloaded_content", "README_CHESTXRAY.pdf")
        
        # Verify file exists
        if not os.path.exists(pdf_path):
            logger.error(f"❌ PDF file not found: {pdf_path}")
            logger.info("Available files in downloaded_content:")
            content_dir = os.path.join(project_root, "downloaded_content")
            if os.path.exists(content_dir):
                for file in os.listdir(content_dir):
                    logger.info(f"   - {file}")
            return
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Process the PDF
        result = await processor.ingest_pdf_as_episodes(pdf_path)
        
        # Print final summary
        processor.print_final_summary(result)
        
        if result['success']:
            logger.info("[SUCCESS] PDF ingestion completed successfully!")
        else:
            logger.error("[ERROR] PDF ingestion failed!")
            
    except Exception as e:
        logger.error(f"[CRITICAL] Critical error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
    finally:
        # Clean up resources
        if processor.service:
            try:
                await processor.service.close()
                logger.info("[CLEANUP] Resources cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    asyncio.run(main())
