#!/usr/bin/env python3
"""
Extract structured content from PDF files and ingest into Neo4j database
Processes both README_CHESTXRAY.pdf and ARXIV_V5_CHESTXRAY.pdf
"""

import os
import sys
import time
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_and_ingest_pdf_files():
    """Extract structured content from PDF files and ingest into Neo4j"""
    
    print("🚀 PDF Structured Content Extraction & Neo4j Ingestion")
    print("=" * 80)
    
    try:
        from neo4j import GraphDatabase
        from core.azure_keyvault_manager import get_secret_from_keyvault
        from data_ingestion.mixed_document import MixedDocumentIngestor
        from data_ingestion.ingestion_structured_document import StructuredDocumentIngestor
        
        # Get Neo4j credentials with validation
        uri = get_secret_from_keyvault("NEO4J_URI")
        username = get_secret_from_keyvault("NEO4J_USERNAME") 
        password = get_secret_from_keyvault("NEO4J_PASSWORD")
        
        # Validate credentials
        if not uri or not username or not password:
            print("❌ Missing Neo4j credentials. Check keyvault/environment configuration.")
            return False
        
        # Initialize components
        driver = GraphDatabase.driver(uri, auth=(username, password))
        structured_ingestor = StructuredDocumentIngestor()
        mixed_ingestor = MixedDocumentIngestor(structured_ingestor=structured_ingestor)
        
        # PDF files to process
        pdf_files = [
            "doc-ingestion/README_CHESTXRAY.pdf",
            "doc-ingestion/ARXIV_V5_CHESTXRAY.pdf"
        ]
        
        # Check if files exist
        existing_files = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                existing_files.append(pdf_file)
                print(f"✅ Found: {pdf_file}")
            else:
                print(f"❌ Missing: {pdf_file}")
        
        if not existing_files:
            print("❌ No PDF files found to process")
            return False
        
        print(f"\n📄 Processing {len(existing_files)} PDF files...")
        
        all_entities = []
        all_relationships = []
        all_structured_data = {}
        total_extraction_time = 0
        
        # Process each PDF file
        for pdf_file in existing_files:
            print(f"\n📑 Processing: {Path(pdf_file).name}")
            print("-" * 60)
            
            start_time = time.time()
            
            try:
                # Extract content using mixed document ingestor
                print(f"🔍 Extracting structured content...")
                extracted_response = mixed_ingestor.ingest_mixed_document(pdf_file)
                
                # Collect extracted data
                file_base = Path(pdf_file).stem
                file_data = {
                    'file_name': Path(pdf_file).name,
                    'file_path': pdf_file,
                    'full_text': extracted_response.full_text,
                    'word_count': len(extracted_response.full_text.split()) if extracted_response.full_text else 0,
                    'chunks_count': len(extracted_response.unstructured_chunks),
                    'entities_count': len(extracted_response.entities),
                    'relationships_count': len(extracted_response.relationships)
                }
                
                all_structured_data[file_base] = file_data
                all_entities.extend(extracted_response.entities)
                all_relationships.extend(extracted_response.relationships)
                
                extraction_time = time.time() - start_time
                total_extraction_time += extraction_time
                
                print(f"   ✅ Extraction complete:")
                print(f"      📊 Word count: {file_data['word_count']:,}")
                print(f"      🧩 Chunks: {file_data['chunks_count']}")
                print(f"      🏷️ Entities: {file_data['entities_count']}")
                print(f"      🔗 Relationships: {file_data['relationships_count']}")
                print(f"      ⏱️ Time: {extraction_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Extraction failed for {pdf_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n📊 Extraction Summary:")
        print(f"   📄 Files processed: {len(all_structured_data)}")
        print(f"   🏷️ Total entities: {len(all_entities)}")
        print(f"   🔗 Total relationships: {len(all_relationships)}")
        print(f"   ⏱️ Total extraction time: {total_extraction_time:.2f}s")
        
        # Now ingest into Neo4j
        print(f"\n💾 Starting Neo4j Ingestion...")
        print("-" * 60)
        
        ingestion_start = time.time()
        
        with driver.session() as session:
            
            # Step 1: Create Document nodes
            print(f"📄 Creating Document nodes...")
            
            document_nodes = []
            for file_base, file_data in all_structured_data.items():
                doc_node = {
                    'document_id': file_base,
                    'file_name': file_data['file_name'],
                    'file_path': file_data['file_path'],
                    'document_type': 'Research Paper',
                    'content_type': 'Mixed (Structured + Unstructured)',
                    'word_count': file_data['word_count'],
                    'chunks_count': file_data['chunks_count'],
                    'entities_count': file_data['entities_count'],
                    'relationships_count': file_data['relationships_count'],
                    'extraction_timestamp': int(time.time())
                }
                document_nodes.append(doc_node)
            
            # Create Document constraint
            try:
                session.run("""
                    CREATE CONSTRAINT unique_document_constraint IF NOT EXISTS
                    FOR (n:Document) REQUIRE n.document_id IS UNIQUE
                """)
                print(f"   ✅ Document constraint created")
            except Exception as e:
                print(f"   ⚠️ Document constraint exists or failed: {e}")
            
            # Batch create Document nodes
            result = session.run("""
                UNWIND $documents AS docData
                CREATE (doc:Document:ResearchPaper)
                SET doc += docData
                RETURN count(doc) as created_count
            """, documents=document_nodes)
            
            record = result.single()
            docs_created = record['created_count'] if record else 0
            print(f"   ✅ Created {docs_created} Document nodes")
            
            # Step 2: Create Entity nodes from extracted entities
            print(f"\n🏷️ Creating Entity nodes...")
            
            if all_entities:
                # Process entities and create unique nodes
                unique_entities = {}
                for entity in all_entities:
                    if isinstance(entity, dict):
                        entity_text = entity.get('text', str(entity))
                        entity_type = entity.get('type', 'Unknown')
                        entity_label = entity.get('label', entity_type)
                    else:
                        entity_text = str(entity)
                        entity_type = 'TextEntity'
                        entity_label = 'TextEntity'
                    
                    entity_key = f"{entity_text}_{entity_type}"
                    if entity_key not in unique_entities:
                        unique_entities[entity_key] = {
                            'entity_text': entity_text,
                            'entity_type': entity_type,
                            'entity_label': entity_label,
                            'occurrence_count': 1
                        }
                    else:
                        unique_entities[entity_key]['occurrence_count'] += 1
                
                entity_nodes = list(unique_entities.values())
                
                # Create Entity constraint
                try:
                    session.run("""
                        CREATE CONSTRAINT unique_entity_constraint IF NOT EXISTS
                        FOR (n:Entity) REQUIRE n.entity_text IS UNIQUE
                    """)
                    print(f"   ✅ Entity constraint created")
                except Exception as e:
                    print(f"   ⚠️ Entity constraint exists or failed: {e}")
                
                # Batch create Entity nodes
                result = session.run("""
                    UNWIND $entities AS entityData
                    CREATE (ent:Entity:TextEntity)
                    SET ent += entityData
                    RETURN count(ent) as created_count
                """, entities=entity_nodes)
                
                record = result.single()
                entities_created = record['created_count'] if record else 0
                print(f"   ✅ Created {entities_created} Entity nodes")
            else:
                entities_created = 0
                print(f"   ℹ️ No entities to create")
            
            # Step 3: Create Concept nodes from structured data
            print(f"\n🧠 Creating Concept nodes...")
            
            concept_nodes = []
            for file_base, file_data in all_structured_data.items():
                # Extract key concepts from the document
                full_text = file_data.get('full_text', '')
                
                # Simple concept extraction (can be enhanced with NLP)
                if 'chest' in full_text.lower():
                    concept_nodes.append({
                        'concept_name': 'Chest X-ray',
                        'concept_type': 'Medical Imaging',
                        'description': 'Radiological examination of the chest',
                        'domain': 'Medical'
                    })
                
                if 'dataset' in full_text.lower():
                    concept_nodes.append({
                        'concept_name': 'Medical Dataset',
                        'concept_type': 'Data Collection',
                        'description': 'Structured collection of medical data',
                        'domain': 'Healthcare Data'
                    })
                
                if 'pathology' in full_text.lower() or 'disease' in full_text.lower():
                    concept_nodes.append({
                        'concept_name': 'Pathology Detection',
                        'concept_type': 'Medical Diagnosis',
                        'description': 'Identification and classification of diseases',
                        'domain': 'Medical Diagnosis'
                    })
            
            # Remove duplicates
            unique_concepts = {}
            for concept in concept_nodes:
                key = concept['concept_name']
                if key not in unique_concepts:
                    unique_concepts[key] = concept
            
            concept_nodes = list(unique_concepts.values())
            
            if concept_nodes:
                # Create Concept constraint
                try:
                    session.run("""
                        CREATE CONSTRAINT unique_concept_constraint IF NOT EXISTS
                        FOR (n:Concept) REQUIRE n.concept_name IS UNIQUE
                    """)
                    print(f"   ✅ Concept constraint created")
                except Exception as e:
                    print(f"   ⚠️ Concept constraint exists or failed: {e}")
                
                # Batch create Concept nodes
                result = session.run("""
                    UNWIND $concepts AS conceptData
                    MERGE (concept:Concept:MedicalConcept)
                    SET concept += conceptData
                    RETURN count(concept) as created_count
                """, concepts=concept_nodes)
                
                record = result.single()
                concepts_created = record['created_count'] if record else 0
                print(f"   ✅ Created {concepts_created} Concept nodes")
            else:
                concepts_created = 0
                print(f"   ℹ️ No concepts to create")
            
            # Step 4: Create relationships
            print(f"\n🔗 Creating Relationships...")
            
            relationships_created = 0
            
            # CONTAINS relationships (Document -> Entity)
            if all_entities and entities_created > 0:
                print(f"   📄 Creating CONTAINS relationships (Document -> Entity)...")
                
                contains_rels = []
                for file_base, file_data in all_structured_data.items():
                    # Create relationships for entities in this document
                    doc_entities = [e for e in all_entities]  # All entities for simplicity
                    
                    for entity in doc_entities[:50]:  # Limit to first 50 entities per document
                        if isinstance(entity, dict):
                            entity_text = entity.get('text', str(entity))
                        else:
                            entity_text = str(entity)
                        
                        contains_rels.append({
                            'document_id': file_base,
                            'entity_text': entity_text
                        })
                
                if contains_rels:
                    result = session.run("""
                        UNWIND $relationships AS rel
                        MATCH (doc:Document {document_id: rel.document_id})
                        MATCH (ent:Entity {entity_text: rel.entity_text})
                        MERGE (doc)-[:CONTAINS]->(ent)
                        RETURN count(*) as created_count
                    """, relationships=contains_rels)
                    
                    record = result.single()
                    contains_count = record['created_count'] if record else 0
                    relationships_created += contains_count
                    print(f"   ✅ Created {contains_count} CONTAINS relationships")
            
            # DESCRIBES relationships (Document -> Concept)
            if concept_nodes:
                print(f"   📖 Creating DESCRIBES relationships (Document -> Concept)...")
                
                describes_rels = []
                for file_base, file_data in all_structured_data.items():
                    for concept in concept_nodes:
                        describes_rels.append({
                            'document_id': file_base,
                            'concept_name': concept['concept_name']
                        })
                
                if describes_rels:
                    result = session.run("""
                        UNWIND $relationships AS rel
                        MATCH (doc:Document {document_id: rel.document_id})
                        MATCH (concept:Concept {concept_name: rel.concept_name})
                        MERGE (doc)-[:DESCRIBES]->(concept)
                        RETURN count(*) as created_count
                    """, relationships=describes_rels)
                    
                    record = result.single()
                    describes_count = record['created_count'] if record else 0
                    relationships_created += describes_count
                    print(f"   ✅ Created {describes_count} DESCRIBES relationships")
            
            # RELATES_TO relationships (Entity -> Concept)
            if entities_created > 0 and concept_nodes:
                print(f"   🔗 Creating RELATES_TO relationships (Entity -> Concept)...")
                
                relates_rels = []
                
                # Simple heuristic-based entity-concept relationships
                for concept in concept_nodes:
                    concept_name = concept['concept_name'].lower()
                    
                    # Find related entities
                    for entity in all_entities[:20]:  # Limit for performance
                        if isinstance(entity, dict):
                            entity_text = entity.get('text', str(entity))
                        else:
                            entity_text = str(entity)
                        
                        entity_lower = entity_text.lower()
                        
                        # Simple relevance matching
                        if ('chest' in concept_name and any(term in entity_lower for term in ['chest', 'lung', 'thorax', 'respiratory'])) or \
                           ('dataset' in concept_name and any(term in entity_lower for term in ['data', 'image', 'patient', 'study'])) or \
                           ('pathology' in concept_name and any(term in entity_lower for term in ['disease', 'pathology', 'diagnosis', 'condition'])):
                            
                            relates_rels.append({
                                'entity_text': entity_text,
                                'concept_name': concept['concept_name']
                            })
                
                if relates_rels:
                    result = session.run("""
                        UNWIND $relationships AS rel
                        MATCH (ent:Entity {entity_text: rel.entity_text})
                        MATCH (concept:Concept {concept_name: rel.concept_name})
                        MERGE (ent)-[:RELATES_TO]->(concept)
                        RETURN count(*) as created_count
                    """, relationships=relates_rels)
                    
                    record = result.single()
                    relates_count = record['created_count'] if record else 0
                    relationships_created += relates_count
                    print(f"   ✅ Created {relates_count} RELATES_TO relationships")
        
        # Calculate timing
        ingestion_time = time.time() - ingestion_start
        total_time = total_extraction_time + ingestion_time
        
        # Final verification
        print(f"\n🎯 Final Database Status:")
        with driver.session() as session:
            
            # Count all nodes by type
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count ORDER BY count DESC")
            total_nodes = 0
            print("   📊 Node counts:")
            for record in result:
                labels = record['labels']
                count = record['count']
                total_nodes += count
                print(f"     {':'.join(labels)}: {count:,}")
            
            # Count all relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC")
            total_rels = 0
            print("   🔗 Relationship counts:")
            for record in result:
                rel_type = record['type']
                count = record['count']
                total_rels += count
                print(f"     {rel_type}: {count:,}")
            
            print(f"   📋 Total nodes: {total_nodes:,}")
            print(f"   🔗 Total relationships: {total_rels:,}")
        
        print(f"\n🎉 SUCCESS: PDF structured content ingestion completed!")
        print(f"   📄 Documents processed: {len(all_structured_data)}")
        print(f"   🏷️ Entities extracted: {len(all_entities)}")
        print(f"   🔗 Relationships created: {relationships_created}")
        print(f"   ⏱️ Extraction time: {total_extraction_time:.2f}s")
        print(f"   💾 Ingestion time: {ingestion_time:.2f}s")
        print(f"   🕐 Total time: {total_time:.2f}s")
        print(f"   🔗 Database: {uri}")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ PDF ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function"""
    
    print("🚀 PDF Mixed Document Extraction & Neo4j Ingestion")
    print("=" * 80)
    
    success = extract_and_ingest_pdf_files()
    
    if success:
        print(f"\n🎉 SUCCESS: PDF files successfully processed and ingested!")
        print(f"\n💡 Try these research queries:")
        print(f"   // Find documents and their entities")
        print(f"   MATCH (doc:Document)-[:CONTAINS]->(ent:Entity)")
        print(f"   RETURN doc.file_name, count(ent) as entity_count")
        print(f"   ORDER BY entity_count DESC")
        print(f"")
        print(f"   // Explore medical concepts")
        print(f"   MATCH (doc:Document)-[:DESCRIBES]->(concept:Concept)")
        print(f"   RETURN concept.concept_name, concept.description, count(doc) as document_count")
        print(f"")
        print(f"   // Find entity-concept relationships")
        print(f"   MATCH (ent:Entity)-[:RELATES_TO]->(concept:Concept)")
        print(f"   RETURN concept.concept_name, collect(ent.entity_text)[..5] as related_entities")
    else:
        print(f"\n❌ FAILED: PDF ingestion was not successful")

if __name__ == "__main__":
    main()
