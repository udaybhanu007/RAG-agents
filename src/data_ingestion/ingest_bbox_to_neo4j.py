#!/usr/bin/env python3
"""
Standalone Neo4j ingestion script for BBox CSV data
This script will actually connect to Neo4j and ingest the data
"""

import os
import sys
import pandas as pd
import json
import re
import time
from typing import Dict, List, Any, Optional
import logging

# Add parent directory to path to access src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from neo4j import GraphDatabase, Driver
    NEO4J_AVAILABLE = True
    print("✅ Neo4j driver available")
except ImportError:
    NEO4J_AVAILABLE = False
    print("❌ Neo4j driver not available")
    sys.exit(1)

from data_ingestion.ingestion_structured_document import StructuredDocumentIngestor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleNeo4jIngestor:
    """Simplified Neo4j ingestor for BBox CSV data"""
    
    def __init__(self, uri: str = "neo4j://localhost:7687", username: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver: Optional[Driver] = None
        self._connect()
    
    def _connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 'Connected to Neo4j!' as message")
                record = result.single()
                message = record["message"] if record else "Connected to Neo4j!"
                print(f"✅ {message}")
                logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            print(f"   URI: {self.uri}")
            print(f"   Username: {self.username}")
            print(f"   Make sure Neo4j is running and credentials are correct")
            raise
    
    def clear_database(self):
        """Clear all existing data"""
        if not self.driver:
            raise ValueError("Driver not initialized")
        print("🗑️ Clearing existing data...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ Database cleared")
    
    def create_constraints(self):
        """Create unique constraints"""
        if not self.driver:
            raise ValueError("Driver not initialized")
        print("🔒 Creating constraints...")
        constraints_created = 0
        
        with self.driver.session() as session:
            # Image constraint
            try:
                session.run("""
                    CREATE CONSTRAINT unique_image_index IF NOT EXISTS
                    FOR (n:Image) REQUIRE n.image_index IS UNIQUE
                """)
                constraints_created += 1
                print("   ✅ Image index constraint created")
            except Exception as e:
                print(f"   ⚠️ Image constraint failed: {e}")
            
            # Finding constraint
            try:
                session.run("""
                    CREATE CONSTRAINT unique_finding_label IF NOT EXISTS
                    FOR (n:Finding) REQUIRE n.finding_label IS UNIQUE
                """)
                constraints_created += 1
                print("   ✅ Finding label constraint created")
            except Exception as e:
                print(f"   ⚠️ Finding constraint failed: {e}")
        
        return constraints_created
    
    def ingest_bbox_csv(self, csv_file: str):
        """Ingest BBox CSV data into Neo4j"""
        print(f"📊 Loading CSV: {csv_file}")
        
        # Load and analyze CSV
        df = pd.read_csv(csv_file)
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Extract data
        unique_images = df['Image Index'].nunique()
        unique_findings = df['Finding Label'].nunique()
        total_relationships = len(df)
        
        print(f"   Unique images: {unique_images}")
        print(f"   Unique findings: {unique_findings}")
        print(f"   Total relationships: {total_relationships}")
        
        # Create Image nodes
        print("\n🖼️ Creating Image nodes...")
        image_nodes = self._create_image_nodes(df)
        
        # Create Finding nodes
        print("🔍 Creating Finding nodes...")
        finding_nodes = self._create_finding_nodes(df)
        
        # Create relationships
        print("🔗 Creating relationships...")
        relationships_created = self._create_relationships(df)
        
        print(f"\n✅ Ingestion completed!")
        print(f"   Images: {image_nodes}")
        print(f"   Findings: {finding_nodes}")
        print(f"   Relationships: {relationships_created}")
        
        return {
            'images': image_nodes,
            'findings': finding_nodes,
            'relationships': relationships_created
        }
    
    def _create_image_nodes(self, df: pd.DataFrame) -> int:
        """Create Image nodes with bounding box data"""
        if not self.driver:
            raise ValueError("Driver not initialized")
        
        # Prepare image data
        image_data = []
        for _, row in df.iterrows():
            if pd.notna(row['Image Index']):
                node_data = {
                    'image_index': str(row['Image Index']),
                    'bbox_x': float(row['Bbox [x']) if pd.notna(row['Bbox [x']) else 0.0,
                    'bbox_y': float(row['y']) if pd.notna(row['y']) else 0.0,
                    'bbox_width': float(row['w']) if pd.notna(row['w']) else 0.0,
                    'bbox_height': float(row['h']) if pd.notna(row['h']) else 0.0,
                }
                image_data.append(node_data)
        
        # Create nodes in batches
        batch_size = 100
        total_created = 0
        
        with self.driver.session() as session:
            for i in range(0, len(image_data), batch_size):
                batch = image_data[i:i+batch_size]
                
                cypher = """
                UNWIND $nodes AS nodeData
                MERGE (img:Image:MedicalImage {image_index: nodeData.image_index})
                SET img += nodeData
                """
                
                session.run(cypher, nodes=batch)
                total_created += len(batch)
                print(f"   Created {total_created}/{len(image_data)} images")
        
        return total_created
    
    def _create_finding_nodes(self, df: pd.DataFrame) -> int:
        """Create Finding nodes"""
        
        # Get unique findings with counts
        finding_counts = df['Finding Label'].value_counts()
        
        finding_data = []
        for finding, count in finding_counts.items():
            node_data = {
                'finding_label': str(finding),
                'occurrence_count': int(count)
            }
            finding_data.append(node_data)
        
        print(f"   Finding distribution:")
        for finding, count in finding_counts.head(8).items():
            print(f"     • {finding}: {count} occurrences")
        
        # Create nodes
        with self.driver.session() as session:
            cypher = """
            UNWIND $nodes AS nodeData
            CREATE (f:Finding:MedicalFinding)
            SET f += nodeData
            """
            
            session.run(cypher, nodes=finding_data)
        
        return len(finding_data)
    
    def _create_relationships(self, df: pd.DataFrame) -> int:
        """Create HAS_FINDING relationships"""
        
        # Prepare relationship data
        relationship_data = []
        for _, row in df.iterrows():
            if pd.notna(row['Image Index']) and pd.notna(row['Finding Label']):
                rel_data = {
                    'image_index': str(row['Image Index']),
                    'finding_label': str(row['Finding Label']),
                    'bbox_x': float(row['Bbox [x']) if pd.notna(row['Bbox [x']) else 0.0,
                    'bbox_y': float(row['y']) if pd.notna(row['y']) else 0.0,
                    'bbox_width': float(row['w']) if pd.notna(row['w']) else 0.0,
                    'bbox_height': float(row['h']) if pd.notna(row['h']) else 0.0,
                }
                relationship_data.append(rel_data)
        
        # Create relationships in batches
        batch_size = 100
        total_created = 0
        
        with self.driver.session() as session:
            for i in range(0, len(relationship_data), batch_size):
                batch = relationship_data[i:i+batch_size]
                
                cypher = """
                UNWIND $relationships AS rel
                MATCH (img:Image {image_index: rel.image_index})
                MATCH (finding:Finding {finding_label: rel.finding_label})
                CREATE (img)-[:HAS_FINDING {
                    bbox_x: rel.bbox_x,
                    bbox_y: rel.bbox_y,
                    bbox_width: rel.bbox_width,
                    bbox_height: rel.bbox_height
                }]->(finding)
                """
                
                session.run(cypher, relationships=batch)
                total_created += len(batch)
                print(f"   Created {total_created}/{len(relationship_data)} relationships")
        
        return total_created
    
    def run_sample_queries(self):
        """Run sample queries to verify the data"""
        print("\n🔍 Running sample queries...")
        
        with self.driver.session() as session:
            # Query 1: Count nodes
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            print("\n📊 Node counts:")
            for record in result:
                labels = record['labels']
                count = record['count']
                print(f"   {labels}: {count}")
            
            # Query 2: Count relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
            print("\n🔗 Relationship counts:")
            for record in result:
                rel_type = record['type']
                count = record['count']
                print(f"   {rel_type}: {count}")
            
            # Query 3: Find images with Atelectasis
            result = session.run("""
                MATCH (img:Image)-[:HAS_FINDING]->(f:Finding {finding_label: 'Atelectasis'})
                RETURN img.image_index
                LIMIT 5
            """)
            print("\n🔍 Sample images with Atelectasis:")
            for record in result:
                print(f"   • {record['img.image_index']}")
            
            # Query 4: Finding distribution
            result = session.run("""
                MATCH (f:Finding)
                RETURN f.finding_label, f.occurrence_count
                ORDER BY f.occurrence_count DESC
            """)
            print("\n📈 Finding distribution:")
            for record in result:
                finding = record['f.finding_label']
                count = record['f.occurrence_count']
                print(f"   • {finding}: {count}")
            
            # Query 5: Large bounding boxes
            result = session.run("""
                MATCH (img:Image)-[r:HAS_FINDING]->(f:Finding)
                WHERE r.bbox_width > 200 AND r.bbox_height > 200
                RETURN img.image_index, f.finding_label, r.bbox_width, r.bbox_height
                ORDER BY r.bbox_width * r.bbox_height DESC
                LIMIT 5
            """)
            print("\n📏 Largest bounding boxes:")
            for record in result:
                image = record['img.image_index']
                finding = record['f.finding_label']
                width = record['r.bbox_width']
                height = record['r.bbox_height']
                area = width * height
                print(f"   • {image}: {finding} ({width:.1f}x{height:.1f} = {area:.0f} pixels)")
    
    def close(self):
        """Close connection"""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j connection closed")

def main():
    """Main ingestion process"""
    print("🚀 BBox CSV to Neo4j Ingestion")
    print("=" * 50)
    
    # Configuration - Update these for your Neo4j setup
    NEO4J_CONFIG = {
        'uri': 'neo4j://localhost:7687',  # Default Neo4j URI
        'username': 'neo4j',              # Default username
        'password': 'password'            # UPDATE THIS!
    }
    
    csv_file = "doc-ingestion/BBox_List_2017.csv"
    
    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    print(f"📁 CSV file: {csv_file}")
    print(f"🔌 Neo4j URI: {NEO4J_CONFIG['uri']}")
    print(f"👤 Username: {NEO4J_CONFIG['username']}")
    
    try:
        # Initialize ingestor
        ingestor = SimpleNeo4jIngestor(**NEO4J_CONFIG)
        
        # Get user confirmation
        response = input("\n⚠️  This will clear the existing database. Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted by user")
            return
        
        start_time = time.time()
        
        # Step 1: Clear database
        ingestor.clear_database()
        
        # Step 2: Create constraints
        constraints = ingestor.create_constraints()
        
        # Step 3: Ingest data
        result = ingestor.ingest_bbox_csv(csv_file)
        
        # Step 4: Run verification queries
        ingestor.run_sample_queries()
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n✅ Ingestion completed in {elapsed:.1f} seconds")
        print(f"📊 Total nodes: {result['images'] + result['findings']}")
        print(f"🔗 Total relationships: {result['relationships']}")
        print(f"🔒 Constraints: {constraints}")
        
        ingestor.close()
        
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure Neo4j is running")
        print("2. Check connection details (URI, username, password)")
        print("3. Verify Neo4j is accessible")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
