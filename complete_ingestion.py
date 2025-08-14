#!/usr/bin/env python3
"""
Complete the ingestion by creating missing Finding nodes and relationships
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.dev')
os.environ['KEYVALUE_ENABLED'] = 'false'

# Add src to path
sys.path.insert(0, 'src')

from src.data_ingestion.neo4j_csv_ingestor import Neo4jCSVIngestor

def create_missing_data():
    """Create missing Finding nodes and Patient-Finding relationships"""
    csv_file = r'doc-ingestion\Data_Entry_2017.csv'
    
    try:
        ingestor = Neo4jCSVIngestor(
            uri=os.getenv('NEO4J_URI'),
            username=os.getenv('NEO4J_USERNAME'), 
            password=os.getenv('NEO4J_PASSWORD')
        )
        
        # Load CSV data
        df = pd.read_csv(csv_file)
        print(f"Loaded CSV with {len(df)} rows")
        
        # Check current state
        stats = ingestor.get_graph_statistics()
        print(f"Current state: {stats}")
        
        # 1. Create Finding nodes manually
        print(f"\n1. Creating Finding nodes...")
        findings = set()
        for _, row in df.iterrows():
            if pd.notna(row['Finding Labels']):
                finding_value = str(row['Finding Labels'])
                if '|' in finding_value:
                    for finding in finding_value.split('|'):
                        finding = finding.strip()
                        if finding and finding != 'No Finding':
                            findings.add(finding)
                else:
                    finding = finding_value.strip()
                    if finding and finding != 'No Finding':
                        findings.add(finding)
        
        # Create Finding nodes
        finding_nodes = []
        for finding in findings:
            count = 0
            for _, row in df.iterrows():
                if pd.notna(row['Finding Labels']):
                    if finding in str(row['Finding Labels']).split('|'):
                        count += 1
            
            finding_nodes.append({
                'name': finding,
                'finding_label': finding,
                'occurrence_count': count
            })
        
        # Batch create Finding nodes
        cypher = """
        UNWIND $nodes AS nodeData
        MERGE (n:Finding {name: nodeData.name})
        SET n += nodeData
        """
        
        with ingestor.driver.session() as session:
            session.run(cypher, nodes=finding_nodes)
        print(f"Created {len(finding_nodes)} Finding nodes")
        
        # 2. Create Patient-Finding relationships
        print(f"\n2. Creating Patient-Finding relationships...")
        cypher = """
        UNWIND $relationships AS rel
        MATCH (p:Patient {id: rel.patient_id})
        MATCH (f:Finding {name: rel.finding_name})
        MERGE (p)-[:HAS_FINDING]->(f)
        """
        
        relationships_data = []
        batch_size = 1000  # Smaller batches
        
        for _, row in df.iterrows():
            if pd.notna(row['Patient ID']) and pd.notna(row['Finding Labels']):
                patient_id = str(row['Patient ID'])
                finding_value = str(row['Finding Labels'])
                
                if '|' in finding_value:
                    for finding in finding_value.split('|'):
                        finding = finding.strip()
                        if finding and finding != 'No Finding':
                            relationships_data.append({
                                'patient_id': patient_id,
                                'finding_name': finding
                            })
                else:
                    finding = finding_value.strip()
                    if finding and finding != 'No Finding':
                        relationships_data.append({
                            'patient_id': patient_id,
                            'finding_name': finding
                        })
        
        # Remove duplicates
        unique_relationships = []
        seen = set()
        for rel in relationships_data:
            key = (rel['patient_id'], rel['finding_name'])
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        print(f"Creating {len(unique_relationships)} unique relationships...")
        
        # Create in batches
        with ingestor.driver.session() as session:
            for i in range(0, len(unique_relationships), batch_size):
                batch = unique_relationships[i:i+batch_size]
                session.run(cypher, relationships=batch)
                print(f"Created batch {i//batch_size + 1}/{(len(unique_relationships) + batch_size - 1)//batch_size}")
        
        # 3. Verify final state
        print(f"\n3. Verification...")
        stats = ingestor.get_graph_statistics()
        print(f"Final state: {stats}")
        
        # Test the query that was failing
        query = "MATCH (p:Patient)-[:HAS_FINDING]->(f:Finding) WHERE p.age < 40 RETURN count(*) as count"
        result = ingestor.query_graph(query)
        print(f"Patients under 40 with findings: {result}")
        
        # Test another query  
        query2 = "MATCH (p:Patient)-[:HAS_FINDING]->(f:Finding) RETURN p.id, p.age, p.gender, f.name LIMIT 5"
        result2 = ingestor.query_graph(query2)
        print(f"Sample data: {result2}")
        
        ingestor.close()
        print(f"\n✅ Ingestion completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_missing_data()
