#!/usr/bin/env python3
"""
Test script to ingest CSV data into Neo4j
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.dev')

# Set keyvault to disabled
os.environ['KEYVALUE_ENABLED'] = 'false'

# Add src to path
sys.path.insert(0, 'src')

from src.data_ingestion.neo4j_csv_ingestor import Neo4jCSVIngestor

def main():
    """Test ingestion"""
    csv_file = r'doc-ingestion\Data_Entry_2017.csv'
    
    print(f"Testing Neo4j connection...")
    print(f"NEO4J_URI: {os.getenv('NEO4J_URI')}")
    print(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME')}")
    print(f"CSV file: {csv_file}")
    
    try:
        # Direct credentials to bypass keyvault issues
        ingestor = Neo4jCSVIngestor(
            uri=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'), 
            password=os.getenv('NEO4J_PASSWORD')
        )
        
        print("Connection successful!")
        
        # Check current state
        stats = ingestor.get_graph_statistics()
        print(f"Current database state:")
        print(f"  Nodes: {stats.get('nodes', {})}")
        print(f"  Relationships: {stats.get('relationships', {})}")
        
        # Run ingestion
        # print(f"\nStarting ingestion...")
        # result = ingestor.ingest_csv_to_neo4j(csv_file, clear_existing=True)
        
        # print(f"\nIngestion Result:")
        # print(f"  Success: {result.success}")
        # print(f"  Nodes Created: {result.nodes_created}")
        # print(f"  Relationships Created: {result.relationships_created}")
        # print(f"  Execution Time: {result.execution_time:.2f}s")
        
        # if result.errors:
        #     print(f"  Errors: {result.errors}")
        
        # # Check final state
        # if result.success:
        stats = ingestor.get_graph_statistics()
        print(f"\nFinal database state:")
        print(f"  Nodes: {stats.get('nodes', {})}")
        print(f"  Relationships: {stats.get('relationships', {})}")
        
        # Test the failing query
        print(f"\nTesting GraphRAG query...")
        query = "MATCH (p:Patient)-[:HAS_FINDING]->(f:Finding) WHERE p.age < 40 RETURN count(*) as count"
        result = ingestor.query_graph(query)
        print(f"Query result: {result}")
        
        #ingestor.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
