#!/usr/bin/env python3
"""
Complete Data_Entry_2017.csv ingestion by adding missing relationships
"""

import os
import sys
import pandas as pd
import time
import logging
from typing import List, Dict, Any

# Add parent directory to path to access src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def complete_data_entry_relationships():
    """Complete the ingestion by adding missing relationships"""
    
    print("🔗 Completing Data_Entry_2017.csv Relationships")
    print("=" * 80)
    
    try:
        from neo4j import GraphDatabase
        from core.azure_keyvault_manager import get_secret_from_keyvault
        
        # Get credentials
        uri = get_secret_from_keyvault("NEO4J_URI")
        username = get_secret_from_keyvault("NEO4J_USERNAME") 
        password = get_secret_from_keyvault("NEO4J_PASSWORD")
        
        # Validate credentials
        if not all([uri, username, password]):
            logger.error("❌ Missing Neo4j credentials")
            return
        
        # Type assertion for mypy
        assert uri is not None and username is not None and password is not None
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Load CSV data
        csv_file = "doc-ingestion/Data_Entry_2017.csv"
        print(f"📊 Loading CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df):,} rows")
        
        # Start relationship creation
        start_time = time.time()
        
        with driver.session() as session:
            
            # Check current relationship status
            print("\n🔍 Checking current relationships...")
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
            current_rels = list(result)
            
            if current_rels:
                print("   Current relationships:")
                for record in current_rels:
                    print(f"     {record['type']}: {record['count']:,}")
            else:
                print("   No relationships found - proceeding with creation")
            
            # Step 1: Create BELONGS_TO relationships (Patient-Image)
            print(f"\n👤 Creating BELONGS_TO relationships (Patient-Image)...")
            
            # Prepare patient-image relationships in batches
            batch_size = 1000
            total_processed = 0
            
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                
                relationships = []
                for _, row in batch_df.iterrows():
                    if pd.notna(row['Image Index']) and pd.notna(row['Patient ID']):
                        rel_data = {
                            'patient_id': int(row['Patient ID']),
                            'image_index': str(row['Image Index']),
                            'followup_number': int(row['Follow-up #']) if pd.notna(row['Follow-up #']) else 0
                        }
                        relationships.append(rel_data)
                
                if relationships:
                    # Create batch of BELONGS_TO relationships
                    result = session.run("""
                        UNWIND $relationships AS rel
                        MATCH (p:Patient {patient_id: rel.patient_id})
                        MATCH (img:Image {image_index: rel.image_index})
                        MERGE (img)-[:BELONGS_TO {followup_number: rel.followup_number}]->(p)
                        RETURN count(*) as created_count
                    """, relationships=relationships)
                    
                    record = result.single()
                    created = record['created_count'] if record else 0
                    total_processed += len(relationships)
                    
                    print(f"   Batch {start_idx//batch_size + 1}: Created {created:,} BELONGS_TO relationships ({total_processed:,}/{len(df):,})")
            
            print(f"✅ Completed BELONGS_TO relationships: {total_processed:,}")
            
            # Step 2: Create HAS_FINDING relationships (Image-Finding)
            print(f"\n🔍 Creating HAS_FINDING relationships (Image-Finding)...")
            
            total_processed = 0
            
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                
                relationships = []
                for _, row in batch_df.iterrows():
                    if pd.notna(row['Image Index']) and pd.notna(row['Finding Labels']):
                        # Handle multiple findings separated by '|'
                        finding_labels = str(row['Finding Labels']).split('|')
                        
                        for finding_label in finding_labels:
                            finding_label = finding_label.strip()
                            if finding_label:
                                rel_data = {
                                    'image_index': str(row['Image Index']),
                                    'finding_label': finding_label,
                                    'view_position': str(row['View Position']) if pd.notna(row['View Position']) else 'Unknown',
                                    'original_width': int(row['OriginalImage[Width']) if pd.notna(row['OriginalImage[Width']) else 0,
                                    'original_height': int(row['Height]']) if pd.notna(row['Height]']) else 0
                                }
                                relationships.append(rel_data)
                
                if relationships:
                    # Create batch of HAS_FINDING relationships
                    result = session.run("""
                        UNWIND $relationships AS rel
                        MATCH (img:Image {image_index: rel.image_index})
                        MATCH (f:Finding {finding_label: rel.finding_label})
                        MERGE (img)-[:HAS_FINDING {
                            view_position: rel.view_position,
                            original_width: rel.original_width,
                            original_height: rel.original_height
                        }]->(f)
                        RETURN count(*) as created_count
                    """, relationships=relationships)
                    
                    record = result.single()
                    created = record['created_count'] if record else 0
                    total_processed += len(relationships)
                    
                    print(f"   Batch {start_idx//batch_size + 1}: Created {created:,} HAS_FINDING relationships ({total_processed:,} total)")
            
            print(f"✅ Completed HAS_FINDING relationships: {total_processed:,}")
            
            # Step 3: Create DIAGNOSED_WITH relationships (Patient-Finding) 
            print(f"\n🏥 Creating DIAGNOSED_WITH relationships (Patient-Finding)...")
            
            # Get patient-finding associations
            patient_findings = {}
            for _, row in df.iterrows():
                if pd.notna(row['Patient ID']) and pd.notna(row['Finding Labels']):
                    patient_id = int(row['Patient ID'])
                    finding_labels = str(row['Finding Labels']).split('|')
                    
                    if patient_id not in patient_findings:
                        patient_findings[patient_id] = {}
                    
                    for finding_label in finding_labels:
                        finding_label = finding_label.strip()
                        if finding_label:
                            if finding_label not in patient_findings[patient_id]:
                                patient_findings[patient_id][finding_label] = 0
                            patient_findings[patient_id][finding_label] += 1
            
            # Create DIAGNOSED_WITH relationships
            relationships = []
            for patient_id, findings in patient_findings.items():
                for finding_label, count in findings.items():
                    rel_data = {
                        'patient_id': patient_id,
                        'finding_label': finding_label,
                        'occurrence_count': count
                    }
                    relationships.append(rel_data)
            
            print(f"   Creating {len(relationships):,} DIAGNOSED_WITH relationships...")
            
            # Process in batches
            total_diagnosed = 0
            for start_idx in range(0, len(relationships), batch_size):
                end_idx = min(start_idx + batch_size, len(relationships))
                batch_rels = relationships[start_idx:end_idx]
                
                result = session.run("""
                    UNWIND $relationships AS rel
                    MATCH (p:Patient {patient_id: rel.patient_id})
                    MATCH (f:Finding {finding_label: rel.finding_label})
                    MERGE (p)-[:DIAGNOSED_WITH {occurrence_count: rel.occurrence_count}]->(f)
                    RETURN count(*) as created_count
                """, relationships=batch_rels)
                
                record = result.single()
                created = record['created_count'] if record else 0
                total_diagnosed += created
                
                print(f"   Batch {start_idx//batch_size + 1}: Created {created:,} DIAGNOSED_WITH relationships")
            
            print(f"✅ Completed DIAGNOSED_WITH relationships: {total_diagnosed:,}")
        
        # Calculate timing
        execution_time = time.time() - start_time
        
        # Final verification
        print(f"\n🎯 Final Verification:")
        with driver.session() as session:
            
            # Count all relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC")
            total_relationships = 0
            print("   Relationship counts:")
            for record in result:
                rel_type = record['type']
                count = record['count']
                total_relationships += count
                print(f"     {rel_type}: {count:,}")
            
            print(f"   Total Relationships: {total_relationships:,}")
            
            # Sample complex queries
            print(f"\n🔍 Sample Complex Queries:")
            
            # Patients with most findings
            result = session.run("""
                MATCH (p:Patient)-[r:DIAGNOSED_WITH]->(f:Finding)
                RETURN p.patient_id, p.gender, p.age, count(f) as finding_count
                ORDER BY finding_count DESC
                LIMIT 5
            """)
            print(f"   Patients with most findings:")
            for record in result:
                print(f"     Patient {record['p.patient_id']} ({record['p.gender']}, {record['p.age']}y): {record['finding_count']} findings")
            
            # Most common finding combinations
            result = session.run("""
                MATCH (img:Image)-[:HAS_FINDING]->(f:Finding)
                WHERE f.finding_label <> 'No Finding'
                WITH img, collect(f.finding_label) as findings
                WHERE size(findings) > 1
                RETURN findings, count(*) as combination_count
                ORDER BY combination_count DESC
                LIMIT 5
            """)
            print(f"   Most common finding combinations:")
            for record in result:
                findings = ', '.join(record['findings'])
                count = record['combination_count']
                print(f"     [{findings}]: {count} images")
        
        print(f"\n🎉 SUCCESS: Data_Entry_2017.csv relationships completed!")
        print(f"⏱️ Total execution time: {execution_time:.2f}s")
        print(f"🔗 Database: {uri}")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Relationship creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function"""
    
    print("🚀 Completing Data_Entry_2017.csv Neo4j Ingestion")
    print("=" * 80)
    
    success = complete_data_entry_relationships()
    
    if success:
        print(f"\n🎉 SUCCESS: Data_Entry_2017.csv ingestion completed!")
        print(f"\n💡 Try these advanced queries:")
        print(f"   // Find patients with multiple pathological findings")
        print(f"   MATCH (p:Patient)-[:DIAGNOSED_WITH]->(f:Finding {{is_normal: false}})")
        print(f"   WITH p, count(f) as pathology_count")
        print(f"   WHERE pathology_count > 1")
        print(f"   RETURN p.patient_id, p.age, p.gender, pathology_count")
        print(f"   ORDER BY pathology_count DESC")
        print(f"")
        print(f"   // Analyze follow-up patterns")
        print(f"   MATCH (img:Image)-[r:BELONGS_TO]->(p:Patient)")
        print(f"   WHERE r.followup_number > 0")
        print(f"   RETURN p.patient_id, count(img) as followup_images")
        print(f"   ORDER BY followup_images DESC")
    else:
        print(f"\n❌ FAILED: Relationship creation was not successful")

if __name__ == "__main__":
    main()
