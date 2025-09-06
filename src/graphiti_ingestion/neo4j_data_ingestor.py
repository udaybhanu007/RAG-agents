#!/usr/bin/env python3
"""
Optimized Neo4j Data Ingestion Script

This script implements best practices for ingesting CSV/Excel data into Neo4j:
- Batch processing for performance
- Schema-driven ingestion using generated schemas
- Relationship inference and creation
- Transaction management
- Error handling and retry logic
- Progress tracking

Author: AI Assistant  
Date: September 2, 2025
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass
import argparse
from neo4j import GraphDatabase, Transaction
from neo4j.exceptions import ServiceUnavailable, TransientError
import time
from tqdm import tqdm
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    neo4j_uri: str
    neo4j_user: str  
    neo4j_password: str
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    create_indexes: bool = True
    enable_constraints: bool = True
    cleanup_on_start: bool = False

@dataclass
class IngestionStats:
    """Statistics for tracking ingestion progress."""
    nodes_created: int = 0
    relationships_created: int = 0
    batches_processed: int = 0
    errors_encountered: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

class Neo4jDataIngestor:
    """
    High-performance Neo4j data ingestion class with best practices.
    """
    
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.driver = None
        self.stats = IngestionStats()
        
    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            logger.info(f"✅ Connected to Neo4j at {self.config.neo4j_uri}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("✅ Neo4j connection closed")
    
    def execute_with_retry(self, session, query: str, parameters: Dict[str, Any] = None) -> Any:
        """Execute query with retry logic for transient errors."""
        for attempt in range(self.config.max_retries):
            try:
                result = session.run(query, parameters or {})
                return result
            except (ServiceUnavailable, TransientError) as e:
                if attempt == self.config.max_retries - 1:
                    raise
                logger.warning(f"Transient error on attempt {attempt + 1}: {e}")
                time.sleep(self.config.retry_delay * (2 ** attempt))
            except Exception as e:
                logger.error(f"Non-transient error: {e}")
                raise
    
    def create_indexes_and_constraints(self, schema: Dict[str, Any]):
        """Create indexes and constraints based on schema analysis."""
        if not self.config.create_indexes:
            return
            
        logger.info("Creating indexes and constraints...")
        
        with self.driver.session() as session:
            try:
                # Create constraints for entity columns (unique identifiers)
                if self.config.enable_constraints:
                    for entity in schema.get('potential_entities', []):
                        constraint_name = f"unique_{entity.lower().replace(' ', '_')}"
                        query = f"""
                        CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                        FOR (n:Entity) REQUIRE n.{self._sanitize_property(entity)} IS UNIQUE
                        """
                        self.execute_with_retry(session, query)
                        logger.info(f"✅ Created constraint for {entity}")
                
                # Create indexes for frequently queried columns
                for column in schema.get('columns', []):
                    if column.get('potential_entity') or column.get('potential_relationship'):
                        index_name = f"idx_{column['name'].lower().replace(' ', '_')}"
                        prop_name = self._sanitize_property(column['name'])
                        query = f"""
                        CREATE INDEX {index_name} IF NOT EXISTS
                        FOR (n:Record) ON (n.{prop_name})
                        """
                        self.execute_with_retry(session, query)
                        logger.info(f"✅ Created index for {column['name']}")
                        
            except Exception as e:
                logger.error(f"❌ Error creating indexes/constraints: {e}")
    
    def cleanup_database(self):
        """Clean up existing data if requested."""
        if not self.config.cleanup_on_start:
            return
            
        logger.warning("🧹 Cleaning up existing data...")
        
        with self.driver.session() as session:
            # Delete all relationships first
            self.execute_with_retry(session, "MATCH ()-[r]-() DELETE r")
            # Delete all nodes
            self.execute_with_retry(session, "MATCH (n) DELETE n")
            logger.info("✅ Database cleaned up")
    
    def _sanitize_property(self, name: str) -> str:
        """Sanitize property names for Neo4j."""
        # Replace spaces and special characters
        sanitized = name.replace(' ', '_').replace('[', '').replace(']', '').replace(',', '_')
        # Remove any other problematic characters
        return ''.join(c for c in sanitized if c.isalnum() or c == '_')
    
    def _generate_node_id(self, row: Dict[str, Any], id_columns: List[str]) -> str:
        """Generate a unique node ID from row data."""
        if id_columns:
            id_parts = [str(row.get(col, '')) for col in id_columns]
            return '_'.join(id_parts)
        else:
            # Fallback: hash the entire row
            row_str = '_'.join(str(v) for v in row.values())
            return hashlib.md5(row_str.encode()).hexdigest()[:16]
    
    def ingest_as_flat_records(self, df: pd.DataFrame, schema: Dict[str, Any]) -> IngestionStats:
        """
        Ingest data as flat records - each row becomes one node.
        Best for: Simple datasets, data exploration, when relationships are unclear.
        """
        logger.info("Starting flat record ingestion...")
        self.stats = IngestionStats()
        self.stats.start_time = datetime.now()
        
        # Prepare data
        df_clean = df.fillna('')  # Replace NaN with empty strings
        total_rows = len(df_clean)
        
        # Create batches
        batches = [df_clean[i:i + self.config.batch_size] 
                  for i in range(0, total_rows, self.config.batch_size)]
        
        logger.info(f"Processing {total_rows} rows in {len(batches)} batches...")
        
        with self.driver.session() as session:
            for batch_idx, batch_df in enumerate(tqdm(batches, desc="Ingesting batches")):
                try:
                    # Prepare batch data
                    records = []
                    for _, row in batch_df.iterrows():
                        record = {
                            'id': self._generate_node_id(row.to_dict(), 
                                                       schema.get('potential_entities', [])),
                            'properties': {self._sanitize_property(k): v 
                                         for k, v in row.to_dict().items()}
                        }
                        records.append(record)
                    
                    # Batch insert query
                    query = """
                    UNWIND $records AS record
                    CREATE (n:Record {id: record.id})
                    SET n += record.properties
                    """
                    
                    self.execute_with_retry(session, query, {'records': records})
                    
                    self.stats.nodes_created += len(records)
                    self.stats.batches_processed += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error in batch {batch_idx}: {e}")
                    self.stats.errors_encountered += 1
        
        self.stats.end_time = datetime.now()
        logger.info(f"✅ Flat record ingestion completed: {self.stats.nodes_created} nodes created")
        return self.stats
    
    def ingest_as_entities_and_relationships(self, df: pd.DataFrame, schema: Dict[str, Any]) -> IngestionStats:
        """
        Ingest data as entities with relationships - normalized approach.
        Best for: Complex datasets, when relationships matter, graph analysis.
        """
        logger.info("Starting entity-relationship ingestion...")
        self.stats = IngestionStats()
        self.stats.start_time = datetime.now()
        
        # Identify entity and relationship columns
        entity_columns = schema.get('potential_entities', [])
        relationship_info = schema.get('potential_relationships', [])
        
        logger.info(f"Entity columns: {entity_columns}")
        logger.info(f"Relationships: {len(relationship_info)}")
        
        df_clean = df.fillna('')
        
        # Step 1: Create entity nodes
        self._create_entity_nodes(df_clean, entity_columns)
        
        # Step 2: Create measurement/attribute nodes
        self._create_attribute_nodes(df_clean, schema)
        
        # Step 3: Create relationships
        self._create_relationships(df_clean, relationship_info, entity_columns)
        
        self.stats.end_time = datetime.now()
        logger.info(f"✅ Entity-relationship ingestion completed")
        return self.stats
    
    def _create_entity_nodes(self, df: pd.DataFrame, entity_columns: List[str]):
        """Create nodes for each unique entity."""
        with self.driver.session() as session:
            for column in entity_columns:
                if column not in df.columns:
                    continue
                    
                unique_values = df[column].unique()
                unique_values = [v for v in unique_values if v != '']  # Remove empty values
                
                logger.info(f"Creating {len(unique_values)} nodes for entity: {column}")
                
                # Create batches of entities
                batches = [unique_values[i:i + self.config.batch_size] 
                          for i in range(0, len(unique_values), self.config.batch_size)]
                
                for batch in tqdm(batches, desc=f"Creating {column} entities"):
                    try:
                        records = [{'value': str(value), 'type': column} for value in batch]
                        
                        query = f"""
                        UNWIND $records AS record
                        MERGE (n:Entity {{value: record.value, type: record.type}})
                        SET n.{self._sanitize_property(column)} = record.value
                        """
                        
                        self.execute_with_retry(session, query, {'records': records})
                        self.stats.nodes_created += len(records)
                        
                    except Exception as e:
                        logger.error(f"❌ Error creating {column} entities: {e}")
                        self.stats.errors_encountered += 1
    
    def _create_attribute_nodes(self, df: pd.DataFrame, schema: Dict[str, Any]):
        """Create nodes for measurements and attributes."""
        with self.driver.session() as session:
            measurement_columns = []
            
            # Identify measurement columns (numeric data, coordinates, etc.)
            for column_info in schema.get('columns', []):
                col_name = column_info['name']
                if (column_info.get('data_type') in ['float', 'integer'] or 
                    'coordinate' in column_info.get('description', '').lower()):
                    measurement_columns.append(col_name)
            
            if not measurement_columns:
                return
                
            logger.info(f"Creating attribute nodes for: {measurement_columns}")
            
            # Process rows in batches
            total_rows = len(df)
            batches = [df[i:i + self.config.batch_size] 
                      for i in range(0, total_rows, self.config.batch_size)]
            
            for batch_df in tqdm(batches, desc="Creating attributes"):
                try:
                    records = []
                    for idx, row in batch_df.iterrows():
                        for col in measurement_columns:
                            if col in row and row[col] != '':
                                records.append({
                                    'row_id': str(idx),
                                    'attribute_name': col,
                                    'attribute_value': str(row[col]),
                                    'data_type': type(row[col]).__name__
                                })
                    
                    if records:
                        query = """
                        UNWIND $records AS record
                        CREATE (a:Attribute {
                            row_id: record.row_id,
                            name: record.attribute_name,
                            value: record.attribute_value,
                            data_type: record.data_type
                        })
                        """
                        
                        self.execute_with_retry(session, query, {'records': records})
                        self.stats.nodes_created += len(records)
                        
                except Exception as e:
                    logger.error(f"❌ Error creating attributes: {e}")
                    self.stats.errors_encountered += 1
    
    def _create_relationships(self, df: pd.DataFrame, relationship_info: List[Dict], entity_columns: List[str]):
        """Create relationships between entities."""
        with self.driver.session() as session:
            total_rows = len(df)
            
            # Create relationships from schema-defined patterns
            for rel_info in relationship_info:
                from_entity = rel_info['from_entity']
                to_entity = rel_info['to_entity']
                rel_type = rel_info['relationship_type']
                
                if from_entity in df.columns and to_entity in df.columns:
                    logger.info(f"Creating {rel_type} relationships: {from_entity} -> {to_entity}")
                    
                    # Process in batches
                    batches = [df[i:i + self.config.batch_size] 
                              for i in range(0, total_rows, self.config.batch_size)]
                    
                    for batch_df in tqdm(batches, desc=f"Creating {rel_type} relationships"):
                        try:
                            relationships = []
                            for _, row in batch_df.iterrows():
                                from_val = str(row[from_entity])
                                to_val = str(row[to_entity])
                                if from_val != '' and to_val != '':
                                    relationships.append({
                                        'from_value': from_val,
                                        'to_value': to_val,
                                        'from_type': from_entity,
                                        'to_type': to_entity
                                    })
                            
                            if relationships:
                                query = f"""
                                UNWIND $relationships AS rel
                                MATCH (from:Entity {{value: rel.from_value, type: rel.from_type}})
                                MATCH (to:Entity {{value: rel.to_value, type: rel.to_type}})
                                MERGE (from)-[r:{rel_type.replace(' ', '_')}]->(to)
                                """
                                
                                self.execute_with_retry(session, query, {'relationships': relationships})
                                self.stats.relationships_created += len(relationships)
                                
                        except Exception as e:
                            logger.error(f"❌ Error creating {rel_type} relationships: {e}")
                            self.stats.errors_encountered += 1
    
    def ingest_data(self, file_path: str, schema_path: str, method: str = 'flat') -> IngestionStats:
        """
        Main ingestion method.
        
        Args:
            file_path: Path to CSV/Excel file
            schema_path: Path to schema JSON file
            method: 'flat' or 'entities' ingestion method
        """
        logger.info(f"Starting data ingestion: {file_path}")
        logger.info(f"Ingestion method: {method}")
        
        # Load data
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Load schema
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        logger.info(f"Loaded schema: {schema['file_name']}")
        
        # Setup database
        self.cleanup_database()
        self.create_indexes_and_constraints(schema)
        
        # Choose ingestion method
        if method == 'flat':
            return self.ingest_as_flat_records(df, schema)
        elif method == 'entities':
            return self.ingest_as_entities_and_relationships(df, schema)
        else:
            raise ValueError(f"Unknown ingestion method: {method}")
    
    def print_ingestion_summary(self, stats: IngestionStats):
        """Print comprehensive ingestion summary."""
        logger.info("\n" + "="*80)
        logger.info("NEO4J INGESTION SUMMARY")
        logger.info("="*80)
        logger.info(f"✅ Status: COMPLETED")
        logger.info(f"📊 Nodes Created: {stats.nodes_created:,}")
        logger.info(f"🔗 Relationships Created: {stats.relationships_created:,}")
        logger.info(f"📦 Batches Processed: {stats.batches_processed}")
        logger.info(f"❌ Errors Encountered: {stats.errors_encountered}")
        logger.info(f"⏱️ Duration: {stats.duration_seconds:.2f} seconds")
        logger.info(f"🚀 Throughput: {stats.nodes_created/stats.duration_seconds:.0f} nodes/sec")
        logger.info("="*80)

def main():
    """Main execution function with command line interface."""
    parser = argparse.ArgumentParser(description="Ingest CSV/Excel data into Neo4j")
    parser.add_argument("input_file", help="Path to CSV/Excel file")
    parser.add_argument("schema_file", help="Path to schema JSON file")
    parser.add_argument("--method", choices=['flat', 'entities'], default='flat',
                       help="Ingestion method (default: flat)")
    parser.add_argument("--neo4j-uri", default="neo4j+s://b2b4aae0.databases.neo4j.io",
                       help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--cleanup", action="store_true", help="Clean database before ingestion")
    parser.add_argument("--no-indexes", action="store_true", help="Skip index creation")
    
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = IngestionConfig(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            batch_size=args.batch_size,
            cleanup_on_start=args.cleanup,
            create_indexes=not args.no_indexes
        )
        
        # Initialize ingestor
        ingestor = Neo4jDataIngestor(config)
        ingestor.connect()
        
        try:
            # Perform ingestion
            stats = ingestor.ingest_data(args.input_file, args.schema_file, args.method)
            
            # Print results
            ingestor.print_ingestion_summary(stats)
            
            return 0
            
        finally:
            ingestor.close()
            
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())
