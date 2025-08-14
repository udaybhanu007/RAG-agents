#!/usr/bin/env python3
"""
Direct PDF content extraction using PyMuPDF and ingestion into Neo4j
Simplified version that doesn't depend on complex imports
"""

import os
import sys
import time
import logging
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to access src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_pdf_content(pdf_path: str) -> Dict[str, Any]:
    """Extract content from PDF file using PyMuPDF"""
    
    print(f"🔍 Extracting content from: {Path(pdf_path).name}")
    
    try:
        doc = fitz.open(pdf_path)
        
        extracted = {
            'file_name': Path(pdf_path).name,
            'file_path': pdf_path,
            'pages_count': len(doc),
            'full_text': '',
            'pages': [],
            'tables': [],
            'sections': [],
            'metadata': {}
        }
        
        full_text = ""
        
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                # Try different PyMuPDF API methods
                if hasattr(page, 'get_text'):
                    page_text = page.get_text()  # type: ignore
                elif hasattr(page, 'getText'):
                    page_text = page.getText()  # type: ignore
                else:
                    # Fallback to string conversion
                    page_text = str(page)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                page_text = ""
            
            page_data = {
                'page_number': page_num + 1,
                'text': page_text,
                'word_count': len(page_text.split())
            }
            
            extracted['pages'].append(page_data)
            full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            # Try to extract tables (if supported)
            try:
                # Check if find_tables method exists (newer PyMuPDF versions)
                if hasattr(page, 'find_tables'):
                    tables = page.find_tables()  # type: ignore
                    if tables and hasattr(tables, 'tables'):
                        for table_idx, table in enumerate(tables.tables):
                            if hasattr(table, 'extract'):
                                table_data = table.extract()
                                if table_data:
                                    extracted['tables'].append({
                                        'page': page_num + 1,
                                'table_index': table_idx,
                                'data': table_data,
                                'rows': len(table_data),
                                'columns': len(table_data[0]) if table_data else 0
                            })
            except Exception as e:
                print(f"   ⚠️ Table extraction failed on page {page_num + 1}: {e}")
        
        extracted['full_text'] = full_text
        extracted['word_count'] = len(full_text.split())
        
        # Extract document metadata
        metadata = doc.metadata or {}  # Handle None metadata
        extracted['metadata'] = {
            'title': metadata.get('title', '') if metadata else '',
            'author': metadata.get('author', '') if metadata else '',
            'subject': metadata.get('subject', '') if metadata else '',
            'creator': metadata.get('creator', '') if metadata else '',
            'producer': metadata.get('producer', '') if metadata else '',
            'creation_date': metadata.get('creationDate', '') if metadata else '',
            'modification_date': metadata.get('modDate', '') if metadata else ''
        }
        
        doc.close()
        
        print(f"   ✅ Extracted {extracted['word_count']:,} words from {extracted['pages_count']} pages")
        print(f"   📊 Found {len(extracted['tables'])} tables")
        
        return extracted
        
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        raise

def extract_entities_from_text(text: str) -> List[Dict[str, Any]]:
    """Simple entity extraction from text using regex patterns"""
    
    entities = []
    
    # Medical terms
    medical_patterns = [
        (r'\b(?:chest|thorax|lung|pulmonary|respiratory|cardiac|heart)\b', 'Anatomy'),
        (r'\b(?:x-ray|radiograph|CT|MRI|ultrasound|imaging|scan)\b', 'ImagingTechnique'),
        (r'\b(?:pneumonia|atelectasis|effusion|cardiomegaly|emphysema|infiltrate|nodule|hernia)\b', 'PathologyFinding'),
        (r'\b(?:patient|subject|case|individual)\b', 'Person'),
        (r'\b(?:dataset|database|collection|study|research|analysis)\b', 'DataCollection'),
        (r'\b(?:diagnosis|detection|classification|screening|evaluation)\b', 'MedicalProcess'),
        (r'\b(?:NIH|hospital|clinic|medical center|institution)\b', 'Organization'),
        (r'\b(?:accuracy|sensitivity|specificity|precision|recall)\b', 'MetricConcept'),
        (r'\b(?:algorithm|model|neural network|deep learning|AI|machine learning)\b', 'Technology')
    ]
    
    for pattern, entity_type in medical_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            entities.append({
                'text': match.group(),
                'type': entity_type,
                'start': match.start(),
                'end': match.end()
            })
    
    # Remove duplicates
    unique_entities = {}
    for entity in entities:
        key = f"{entity['text'].lower()}_{entity['type']}"
        if key not in unique_entities:
            unique_entities[key] = entity
    
    return list(unique_entities.values())

def extract_concepts_from_content(extracted_content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract key concepts from the extracted content"""
    
    concepts = []
    text = extracted_content['full_text'].lower()
    file_name = extracted_content['file_name'].lower()
    
    # Medical imaging concepts
    if any(term in text for term in ['chest x-ray', 'radiograph', 'imaging']):
        concepts.append({
            'concept_name': 'Chest X-ray Imaging',
            'concept_type': 'Medical Imaging',
            'description': 'Radiological examination of the chest using X-rays',
            'domain': 'Radiology',
            'confidence': 0.9
        })
    
    # Dataset concepts
    if any(term in text for term in ['dataset', 'data collection', 'database']):
        concepts.append({
            'concept_name': 'Medical Dataset',
            'concept_type': 'Data Collection',
            'description': 'Structured collection of medical data for research',
            'domain': 'Healthcare Data',
            'confidence': 0.8
        })
    
    # AI/ML concepts
    if any(term in text for term in ['machine learning', 'deep learning', 'neural network', 'algorithm']):
        concepts.append({
            'concept_name': 'Medical AI',
            'concept_type': 'Artificial Intelligence',
            'description': 'Application of AI techniques in medical diagnosis',
            'domain': 'Healthcare Technology',
            'confidence': 0.8
        })
    
    # Pathology concepts
    if any(term in text for term in ['pathology', 'disease', 'diagnosis', 'finding']):
        concepts.append({
            'concept_name': 'Pathology Detection',
            'concept_type': 'Medical Diagnosis',
            'description': 'Identification and classification of diseases',
            'domain': 'Medical Diagnosis',
            'confidence': 0.9
        })
    
    # Research concepts
    if any(term in text for term in ['research', 'study', 'analysis', 'evaluation']):
        concepts.append({
            'concept_name': 'Medical Research',
            'concept_type': 'Scientific Research',
            'description': 'Scientific investigation in medical domain',
            'domain': 'Medical Research',
            'confidence': 0.7
        })
    
    return concepts

def ingest_pdf_to_neo4j():
    """Extract PDF content and ingest into Neo4j"""
    
    print("🚀 Direct PDF Content Extraction & Neo4j Ingestion")
    print("=" * 80)
    
    try:
        from neo4j import GraphDatabase
        from core.azure_keyvault_manager import get_secret_from_keyvault
        
        # Get Neo4j credentials
        uri = get_secret_from_keyvault("NEO4J_URI")
        username = get_secret_from_keyvault("NEO4J_USERNAME") 
        password = get_secret_from_keyvault("NEO4J_PASSWORD")
        
        # Validate credentials
        if not all([uri, username, password]):
            logger.error("❌ Missing Neo4j credentials")
            return
        
        # Type assertion for mypy
        assert uri is not None and username is not None and password is not None
        
        # Initialize Neo4j connection
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
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
        
        all_documents = []
        all_entities = []
        all_concepts = []
        total_extraction_time = 0
        
        # Process each PDF file
        for pdf_file in existing_files:
            print(f"\n📑 Processing: {Path(pdf_file).name}")
            print("-" * 60)
            
            start_time = time.time()
            
            try:
                # Extract content
                extracted_content = extract_pdf_content(pdf_file)
                
                # Extract entities
                entities = extract_entities_from_text(extracted_content['full_text'])
                
                # Extract concepts
                concepts = extract_concepts_from_content(extracted_content)
                
                # Prepare document data
                doc_data = {
                    'document_id': Path(pdf_file).stem,
                    'file_name': extracted_content['file_name'],
                    'file_path': extracted_content['file_path'],
                    'document_type': 'Research Paper',
                    'content_type': 'Mixed PDF Document',
                    'pages_count': extracted_content['pages_count'],
                    'word_count': extracted_content['word_count'],
                    'tables_count': len(extracted_content['tables']),
                    'entities_count': len(entities),
                    'concepts_count': len(concepts),
                    'title': extracted_content['metadata'].get('title', ''),
                    'author': extracted_content['metadata'].get('author', ''),
                    'subject': extracted_content['metadata'].get('subject', ''),
                    'extraction_timestamp': int(time.time())
                }
                
                all_documents.append(doc_data)
                
                # Add document reference to entities and concepts
                for entity in entities:
                    entity['document_id'] = doc_data['document_id']
                    all_entities.append(entity)
                
                for concept in concepts:
                    concept['document_id'] = doc_data['document_id']
                    all_concepts.append(concept)
                
                extraction_time = time.time() - start_time
                total_extraction_time += extraction_time
                
                print(f"   ✅ Extraction complete:")
                print(f"      📊 Word count: {doc_data['word_count']:,}")
                print(f"      📄 Pages: {doc_data['pages_count']}")
                print(f"      📊 Tables: {doc_data['tables_count']}")
                print(f"      🏷️ Entities: {doc_data['entities_count']}")
                print(f"      🧠 Concepts: {doc_data['concepts_count']}")
                print(f"      ⏱️ Time: {extraction_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Extraction failed for {pdf_file}: {e}")
                continue
        
        print(f"\n📊 Extraction Summary:")
        print(f"   📄 Documents processed: {len(all_documents)}")
        print(f"   🏷️ Total entities: {len(all_entities)}")
        print(f"   🧠 Total concepts: {len(all_concepts)}")
        print(f"   ⏱️ Total extraction time: {total_extraction_time:.2f}s")
        
        # Now ingest into Neo4j
        print(f"\n💾 Starting Neo4j Ingestion...")
        print("-" * 60)
        
        ingestion_start = time.time()
        
        with driver.session() as session:
            
            # Step 1: Create Document nodes
            print(f"📄 Creating Document nodes...")
            
            # Create Document constraint
            try:
                session.run("""
                    CREATE CONSTRAINT unique_pdf_document_constraint IF NOT EXISTS
                    FOR (n:PDFDocument) REQUIRE n.document_id IS UNIQUE
                """)
                print(f"   ✅ PDFDocument constraint created")
            except Exception as e:
                print(f"   ⚠️ PDFDocument constraint exists or failed: {e}")
            
            # Batch create Document nodes
            result = session.run("""
                UNWIND $documents AS docData
                CREATE (doc:PDFDocument:ResearchPaper)
                SET doc += docData
                RETURN count(doc) as created_count
            """, documents=all_documents)
            
            record = result.single()
            docs_created = record['created_count'] if record else 0
            print(f"   ✅ Created {docs_created} PDFDocument nodes")
            
            # Step 2: Create Entity nodes
            print(f"\n🏷️ Creating Entity nodes...")
            
            if all_entities:
                # Group entities by type and text to avoid duplicates
                unique_entities = {}
                for entity in all_entities:
                    key = f"{entity['text']}_{entity['type']}"
                    if key not in unique_entities:
                        unique_entities[key] = {
                            'entity_text': entity['text'],
                            'entity_type': entity['type'],
                            'occurrence_count': 1,
                            'documents': [entity['document_id']]
                        }
                    else:
                        unique_entities[key]['occurrence_count'] += 1
                        if entity['document_id'] not in unique_entities[key]['documents']:
                            unique_entities[key]['documents'].append(entity['document_id'])
                
                entity_nodes = []
                for entity_data in unique_entities.values():
                    entity_nodes.append({
                        'entity_text': entity_data['entity_text'],
                        'entity_type': entity_data['entity_type'],
                        'occurrence_count': entity_data['occurrence_count'],
                        'document_count': len(entity_data['documents'])
                    })
                
                # Create Entity constraint
                try:
                    session.run("""
                        CREATE CONSTRAINT unique_pdf_entity_constraint IF NOT EXISTS
                        FOR (n:PDFEntity) REQUIRE (n.entity_text, n.entity_type) IS UNIQUE
                    """)
                    print(f"   ✅ PDFEntity constraint created")
                except Exception as e:
                    print(f"   ⚠️ PDFEntity constraint exists or failed: {e}")
                
                # Batch create Entity nodes
                result = session.run("""
                    UNWIND $entities AS entityData
                    CREATE (ent:PDFEntity:MedicalEntity)
                    SET ent += entityData
                    RETURN count(ent) as created_count
                """, entities=entity_nodes)
                
                record = result.single()
                entities_created = record['created_count'] if record else 0
                print(f"   ✅ Created {entities_created} PDFEntity nodes")
            else:
                entities_created = 0
                print(f"   ℹ️ No entities to create")
            
            # Step 3: Create Concept nodes
            print(f"\n🧠 Creating Concept nodes...")
            
            if all_concepts:
                # Group concepts to avoid duplicates
                unique_concepts = {}
                for concept in all_concepts:
                    key = concept['concept_name']
                    if key not in unique_concepts:
                        unique_concepts[key] = {
                            'concept_name': concept['concept_name'],
                            'concept_type': concept['concept_type'],
                            'description': concept['description'],
                            'domain': concept['domain'],
                            'confidence': concept['confidence'],
                            'documents': [concept['document_id']]
                        }
                    else:
                        if concept['document_id'] not in unique_concepts[key]['documents']:
                            unique_concepts[key]['documents'].append(concept['document_id'])
                
                concept_nodes = []
                for concept_data in unique_concepts.values():
                    concept_nodes.append({
                        'concept_name': concept_data['concept_name'],
                        'concept_type': concept_data['concept_type'],
                        'description': concept_data['description'],
                        'domain': concept_data['domain'],
                        'confidence': concept_data['confidence'],
                        'document_count': len(concept_data['documents'])
                    })
                
                # Create Concept constraint
                try:
                    session.run("""
                        CREATE CONSTRAINT unique_pdf_concept_constraint IF NOT EXISTS
                        FOR (n:PDFConcept) REQUIRE n.concept_name IS UNIQUE
                    """)
                    print(f"   ✅ PDFConcept constraint created")
                except Exception as e:
                    print(f"   ⚠️ PDFConcept constraint exists or failed: {e}")
                
                # Batch create Concept nodes
                result = session.run("""
                    UNWIND $concepts AS conceptData
                    CREATE (concept:PDFConcept:MedicalConcept)
                    SET concept += conceptData
                    RETURN count(concept) as created_count
                """, concepts=concept_nodes)
                
                record = result.single()
                concepts_created = record['created_count'] if record else 0
                print(f"   ✅ Created {concepts_created} PDFConcept nodes")
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
                for entity in all_entities:
                    contains_rels.append({
                        'document_id': entity['document_id'],
                        'entity_text': entity['text'],
                        'entity_type': entity['type']
                    })
                
                # Remove duplicates
                unique_contains = {}
                for rel in contains_rels:
                    key = f"{rel['document_id']}_{rel['entity_text']}_{rel['entity_type']}"
                    if key not in unique_contains:
                        unique_contains[key] = rel
                
                contains_rels = list(unique_contains.values())
                
                if contains_rels:
                    result = session.run("""
                        UNWIND $relationships AS rel
                        MATCH (doc:PDFDocument {document_id: rel.document_id})
                        MATCH (ent:PDFEntity {entity_text: rel.entity_text, entity_type: rel.entity_type})
                        MERGE (doc)-[:CONTAINS]->(ent)
                        RETURN count(*) as created_count
                    """, relationships=contains_rels)
                    
                    record = result.single()
                    contains_count = record['created_count'] if record else 0
                    relationships_created += contains_count
                    print(f"   ✅ Created {contains_count} CONTAINS relationships")
            
            # DESCRIBES relationships (Document -> Concept)
            if all_concepts and concepts_created > 0:
                print(f"   📖 Creating DESCRIBES relationships (Document -> Concept)...")
                
                describes_rels = []
                for concept in all_concepts:
                    describes_rels.append({
                        'document_id': concept['document_id'],
                        'concept_name': concept['concept_name']
                    })
                
                # Remove duplicates
                unique_describes = {}
                for rel in describes_rels:
                    key = f"{rel['document_id']}_{rel['concept_name']}"
                    if key not in unique_describes:
                        unique_describes[key] = rel
                
                describes_rels = list(unique_describes.values())
                
                if describes_rels:
                    result = session.run("""
                        UNWIND $relationships AS rel
                        MATCH (doc:PDFDocument {document_id: rel.document_id})
                        MATCH (concept:PDFConcept {concept_name: rel.concept_name})
                        MERGE (doc)-[:DESCRIBES]->(concept)
                        RETURN count(*) as created_count
                    """, relationships=describes_rels)
                    
                    record = result.single()
                    describes_count = record['created_count'] if record else 0
                    relationships_created += describes_count
                    print(f"   ✅ Created {describes_count} DESCRIBES relationships")
            
            # RELATES_TO relationships (Entity -> Concept)
            if entities_created > 0 and concepts_created > 0:
                print(f"   🔗 Creating RELATES_TO relationships (Entity -> Concept)...")
                
                relates_rels = []
                
                # Create entity-concept relationships based on domain matching
                for entity in all_entities:
                    entity_type = entity['type'].lower()
                    entity_text = entity['text'].lower()
                    
                    for concept in all_concepts:
                        concept_name = concept['concept_name'].lower()
                        concept_type = concept['concept_type'].lower()
                        concept_domain = concept['domain'].lower()
                        
                        # Simple relevance matching
                        should_relate = False
                        
                        if 'anatomy' in entity_type and 'imaging' in concept_name:
                            should_relate = True
                        elif 'imagingtechnique' in entity_type and 'imaging' in concept_name:
                            should_relate = True
                        elif 'pathologyfinding' in entity_type and 'pathology' in concept_name:
                            should_relate = True
                        elif 'datacollection' in entity_type and 'dataset' in concept_name:
                            should_relate = True
                        elif 'technology' in entity_type and 'ai' in concept_name:
                            should_relate = True
                        elif 'medicalprocess' in entity_type and 'diagnosis' in concept_name:
                            should_relate = True
                        
                        if should_relate:
                            relates_rels.append({
                                'entity_text': entity['text'],
                                'entity_type': entity['type'],
                                'concept_name': concept['concept_name']
                            })
                
                # Remove duplicates
                unique_relates = {}
                for rel in relates_rels:
                    key = f"{rel['entity_text']}_{rel['entity_type']}_{rel['concept_name']}"
                    if key not in unique_relates:
                        unique_relates[key] = rel
                
                relates_rels = list(unique_relates.values())
                
                if relates_rels:
                    result = session.run("""
                        UNWIND $relationships AS rel
                        MATCH (ent:PDFEntity {entity_text: rel.entity_text, entity_type: rel.entity_type})
                        MATCH (concept:PDFConcept {concept_name: rel.concept_name})
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
            
            # Count PDF-related nodes
            result = session.run("""
                MATCH (n) 
                WHERE 'PDFDocument' IN labels(n) OR 'PDFEntity' IN labels(n) OR 'PDFConcept' IN labels(n)
                RETURN labels(n) as labels, count(n) as count 
                ORDER BY count DESC
            """)
            print("   📊 PDF Node counts:")
            for record in result:
                labels = record['labels']
                count = record['count']
                print(f"     {':'.join(labels)}: {count:,}")
            
            # Count PDF-related relationships
            result = session.run("""
                MATCH (n)-[r]->(m) 
                WHERE ('PDFDocument' IN labels(n) OR 'PDFEntity' IN labels(n) OR 'PDFConcept' IN labels(n))
                   OR ('PDFDocument' IN labels(m) OR 'PDFEntity' IN labels(m) OR 'PDFConcept' IN labels(m))
                RETURN type(r) as type, count(r) as count 
                ORDER BY count DESC
            """)
            print("   🔗 PDF Relationship counts:")
            for record in result:
                rel_type = record['type']
                count = record['count']
                print(f"     {rel_type}: {count:,}")
        
        print(f"\n🎉 SUCCESS: PDF structured content ingestion completed!")
        print(f"   📄 Documents processed: {len(all_documents)}")
        print(f"   🏷️ Entities extracted: {len(all_entities)}")
        print(f"   🧠 Concepts extracted: {len(all_concepts)}")
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
    
    print("🚀 Direct PDF Content Extraction & Neo4j Ingestion")
    print("=" * 80)
    
    success = ingest_pdf_to_neo4j()
    
    if success:
        print(f"\n🎉 SUCCESS: PDF files successfully processed and ingested!")
        print(f"\n💡 Try these research queries:")
        print(f"   // Find PDF documents and their entities")
        print(f"   MATCH (doc:PDFDocument)-[:CONTAINS]->(ent:PDFEntity)")
        print(f"   RETURN doc.file_name, ent.entity_type, count(ent) as entity_count")
        print(f"   ORDER BY entity_count DESC")
        print(f"")
        print(f"   // Explore medical concepts in PDFs")
        print(f"   MATCH (doc:PDFDocument)-[:DESCRIBES]->(concept:PDFConcept)")
        print(f"   RETURN concept.concept_name, concept.description, concept.domain")
        print(f"")
        print(f"   // Find entity-concept relationships from PDFs")
        print(f"   MATCH (ent:PDFEntity)-[:RELATES_TO]->(concept:PDFConcept)")
        print(f"   RETURN concept.concept_name, concept.domain, collect(ent.entity_text)[..5] as related_entities")
    else:
        print(f"\n❌ FAILED: PDF ingestion was not successful")

if __name__ == "__main__":
    main()
