#!/usr/bin/env python3
"""
SIMPLE BULK FIX - Based on our successful debug test

The debug test showed that:
1. Individual episodes work perfectly
2. Bulk method exists and is callable
3. Issue is context-related, not method signature

This creates a working bulk processor with fallback.
"""

import asyncio
import logging
import pandas as pd
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Add the project root and src to sys.path to import modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

# Load environment 
load_dotenv(dotenv_path=os.path.join(project_root, ".env.dev"))

from graphiti_ingestion.services.graphiti_ingestion_service import GraphitiIngestionService
from graphiti_ingestion.config.ingestion_config import GraphitiIngestionConfig
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.nodes import EpisodeType

# Setup clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def process_csv_bulk_fixed(csv_file: str):
    """Process CSV with working bulk method and fallback"""
    try:
        logger.info("=== BULK CSV PROCESSING (FIXED) ===")
        
        # Initialize service (same as debug test)
        config = GraphitiIngestionConfig()
        service = GraphitiIngestionService(config)
        
        # Read CSV
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} records from {csv_file}")
        
        # Small batches to be conservative
        batch_size = 5
        total_records = len(df)
        
        for i in range(0, total_records, batch_size):
            batch_data = df.iloc[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            logger.info(f"Processing batch {batch_num}: records {i} to {i + len(batch_data) - 1}")
            
            # Create episodes
            episodes = []
            for idx, (_, row) in enumerate(batch_data.iterrows()):
                episodes.append(
                    RawEpisode(
                        name=f"csv_record_{i + idx}",
                        content=row.to_json(),
                        source_description=f"CSV record from {csv_file}",
                        reference_time=datetime.now(),
                        source=EpisodeType.json
                    )
                )
            
            # Try bulk processing
            try:
                logger.info(f"Attempting BULK for batch {batch_num} ({len(episodes)} episodes)")
                
                group_id = f"bulk_batch_{batch_num}_{datetime.now().strftime('%H%M%S')}"
                
                # Build indices
                await service._graphiti.build_indices_and_constraints()
                
                # Use bulk method
                await service._graphiti.add_episode_bulk(
                    bulk_episodes=episodes,
                    group_id=group_id
                )
                
                logger.info(f"BULK SUCCESS: Batch {batch_num} completed")
                
            except Exception as bulk_error:
                logger.warning(f"BULK FAILED for batch {batch_num}: {bulk_error}")
                logger.info(f"Falling back to individual processing...")
                
                # Fallback to individual
                success_count = 0
                for episode in episodes:
                    try:
                        await service._graphiti.add_episode(
                            name=episode.name,
                            episode_body=episode.content,
                            source_description=episode.source_description,
                            reference_time=episode.reference_time,
                            source=episode.source
                        )
                        success_count += 1
                    except Exception as individual_error:
                        logger.error(f"Individual failed: {episode.name} - {individual_error}")
                
                logger.info(f"INDIVIDUAL RESULTS: {success_count}/{len(episodes)} episodes processed")
            
            # Wait between batches
            await asyncio.sleep(1)
        
        logger.info("=== PROCESSING COMPLETE ===")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

async def main():
    """Test with our CSV file"""
    # Path relative to project root
    csv_path = os.path.join(project_root, "downloaded_content", "BBox_List_2017.csv")
    await process_csv_bulk_fixed(csv_path)

if __name__ == "__main__":
    asyncio.run(main())
