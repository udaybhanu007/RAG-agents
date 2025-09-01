#!/usr/bin/env python3
"""
Simplified Production Bulk CSV Processor with Adaptive Batch Sizing
==================================================================

This script processes CSV data using adaptive batch sizing for optimal performance.

Key Features:
- Adaptive batch sizing based on success rates
- Automatic fallback to individual processing when bulk fails
- Simple and maintainable code structure
"""

import asyncio
import pandas as pd
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any
import sys
import os

# Add the project root and src to sys.path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

try:
    from graphiti_ingestion.services.graphiti_ingestion_service import GraphitiIngestionService
    from graphiti_ingestion.config.ingestion_config import GraphitiIngestionConfig
    from graphiti_core.utils.bulk_utils import RawEpisode
    from graphiti_core.nodes import EpisodeType
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have the required dependencies installed")
    sys.exit(1)

# Simple logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaptiveBatchProcessor:
    """Simple adaptive batch sizing based on success rates."""
    
    def __init__(self, initial_size: int = 5, min_size: int = 2, max_size: int = 10):
        self.batch_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.recent_results = []  # Track last 10 results
        
    def record_result(self, success: bool):
        """Record batch processing result."""
        self.recent_results.append(success)
        if len(self.recent_results) > 10:
            self.recent_results.pop(0)
    
    def adjust_batch_size(self):
        """Adjust batch size based on recent success rate."""
        if len(self.recent_results) < 3:
            return  # Need minimum data
            
        success_rate = sum(self.recent_results) / len(self.recent_results)
        
        if success_rate >= 0.8 and self.batch_size < self.max_size:
            self.batch_size += 1
        elif success_rate <= 0.4 and self.batch_size > self.min_size:
            self.batch_size -= 1
    
    def get_batch_size(self) -> int:
        return self.batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.recent_results:
            return {}
        success_rate = sum(self.recent_results) / len(self.recent_results)
        return {
            'success_rate': f"{success_rate:.1%}",
            'current_batch_size': self.batch_size,
            'total_attempts': len(self.recent_results)
        }

class ProductionBulkProcessor:
    """Simplified bulk processor with adaptive batch sizing."""
    
    def __init__(self):
        self.service = None
        self.adaptive_processor = AdaptiveBatchProcessor()
        self.stats = {
            'total_processed': 0,
            'bulk_success': 0,
            'individual_fallback': 0,
            'total_time': 0.0,
            'bulk_time': 0.0,
            'individual_time': 0.0
        }
    
    async def initialize_service(self):
        """Initialize Graphiti service."""
        try:
            env_path = os.path.join(project_root, '.env.dev')
            
            if not os.path.exists(env_path):
                logger.error(f"Environment file not found: {env_path}")
                raise FileNotFoundError(f"Environment file not found: {env_path}")
            
            load_dotenv(dotenv_path=env_path, override=True)
            
            # Validate required environment variables
            required_vars = ['NEO4J_URI', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_DEPLOYMENT']
            for var in required_vars:
                if not os.getenv(var):
                    raise ValueError(f"{var} environment variable is not set or empty")
            
            config = GraphitiIngestionConfig()
            self.service = GraphitiIngestionService(config)
            
            logger.info("✅ Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize service: {e}")
            raise
    
    async def process_batch_bulk(self, batch_episodes: List[RawEpisode]) -> bool:
        """Try bulk processing for a batch of episodes."""
        try:
            start_time = time.time()
            
            await self.service._graphiti.build_indices_and_constraints()
            await self.service._graphiti.add_episode_bulk(
                bulk_episodes=batch_episodes,
                group_id="csv_bulk_production"
            )
            
            elapsed = time.time() - start_time
            self.stats['bulk_success'] += len(batch_episodes)
            self.stats['bulk_time'] += elapsed
            
            # Record success
            self.adaptive_processor.record_result(True)
            
            logger.info(f"✅ Bulk processed {len(batch_episodes)} episodes in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            # Record failure
            self.adaptive_processor.record_result(False)
            logger.warning(f"Bulk failed, using individual fallback: {type(e).__name__}")
            return False
    
    async def process_individual_fallback(self, episodes: List[RawEpisode]) -> None:
        """Process episodes individually when bulk fails."""
        start_time = time.time()
        
        for i, episode in enumerate(episodes):
            try:
                await self.service._graphiti.add_episode(
                    name=episode.name,
                    episode_body=episode.content,
                    source_description=episode.source_description,
                    reference_time=episode.reference_time,
                    source=episode.source
                )
                await asyncio.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Individual episode {i+1} failed: {e}")
        
        elapsed = time.time() - start_time
        self.stats['individual_fallback'] += len(episodes)
        self.stats['individual_time'] += elapsed
        logger.info(f"Individual fallback completed {len(episodes)} episodes in {elapsed:.2f}s")
    
    async def process_csv_production(self, csv_path: str, initial_batch_size: int = 5) -> None:
        """Production CSV processing with adaptive batch sizing."""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Processing {len(df)} records from CSV")
            
            # Convert to RawEpisode objects
            episodes = []
            for idx, row in df.iterrows():
                episode = RawEpisode(
                    name=f"BBox_Entry_{idx}",
                    content=f"Medical imaging data - Finding: {row.get('Finding Label', 'Unknown')}, "
                            f"X: {row.get('Bbox [x', 'N/A')}, Y: {row.get('y', 'N/A')}, "
                            f"Width: {row.get('w', 'N/A')}, Height: {row.get('h]', 'N/A')}",
                    source_description="BBox chest X-ray findings dataset with bounding box coordinates",
                    source=EpisodeType.json,
                    reference_time=datetime.now(timezone.utc)
                )
                episodes.append(episode)
            
            logger.info(f"📋 Processing {len(episodes)} episodes with adaptive batching")
            
            # Initialize adaptive processor
            self.adaptive_processor = AdaptiveBatchProcessor(
                initial_size=initial_batch_size,
                min_size=2,
                max_size=15
            )
            
            # Process with adaptive batching
            total_start = time.time()
            processed = 0
            
            while processed < len(episodes):
                # Get current batch size
                current_batch_size = self.adaptive_processor.get_batch_size()
                
                # Create batch
                batch = episodes[processed:processed + current_batch_size]
                
                # Try bulk first
                bulk_success = await self.process_batch_bulk(batch)
                
                if not bulk_success:
                    # Fallback to individual processing
                    await self.process_individual_fallback(batch)
                
                processed += len(batch)
                self.stats['total_processed'] += len(batch)
                
                # Adjust batch size based on results
                self.adaptive_processor.adjust_batch_size()
                
                # Brief progress update
                if processed % 10 == 0 or processed == len(episodes):
                    progress = (processed / len(episodes)) * 100
                    logger.info(f"Progress: {processed}/{len(episodes)} ({progress:.1f}%)")
                
                # Short delay between batches
                await asyncio.sleep(1.0)
            
            # Final statistics
            total_time = time.time() - total_start
            self.stats['total_time'] = total_time
            
            self.print_final_stats()
            
        except Exception as e:
            logger.error(f"❌ CSV processing failed: {e}")
            raise
    
    def print_final_stats(self):
        """Print processing statistics."""
        stats = self.stats
        
        logger.info("\n" + "="*50)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*50)
        
        # Overall stats
        logger.info(f"Episodes Processed: {stats['total_processed']}")
        logger.info(f"Total Time: {stats['total_time']:.2f}s")
        logger.info(f"Average per Episode: {stats['total_time']/stats['total_processed']:.2f}s")
        
        # Method breakdown
        if stats['bulk_success'] > 0:
            bulk_pct = (stats['bulk_success'] / stats['total_processed']) * 100
            logger.info(f"Bulk Processing: {stats['bulk_success']} episodes ({bulk_pct:.1f}%)")
        
        if stats['individual_fallback'] > 0:
            individual_pct = (stats['individual_fallback'] / stats['total_processed']) * 100
            logger.info(f"Individual Processing: {stats['individual_fallback']} episodes ({individual_pct:.1f}%)")
        
        # Final adaptive state
        adaptive_stats = self.adaptive_processor.get_stats()
        if adaptive_stats:
            logger.info(f"Final Success Rate: {adaptive_stats['success_rate']}")
            logger.info(f"Final Batch Size: {adaptive_stats['current_batch_size']}")
        
        logger.info("="*50)

async def main():
    """Main execution function."""
    processor = ProductionBulkProcessor()
    
    try:
        await processor.initialize_service()
        
        # Process the CSV with production settings
        csv_path = os.path.join(project_root, "downloaded_content", "BBox_List_2017.csv")
        await processor.process_csv_production(
            csv_path=csv_path,
            initial_batch_size=5  # Starting batch size for adaptive optimization
        )
        
    except Exception as e:
        logger.error(f"❌ Production processing failed: {e}")
        raise
    
    finally:
        if processor.service:
            await processor.service.close()

if __name__ == "__main__":
    asyncio.run(main())
