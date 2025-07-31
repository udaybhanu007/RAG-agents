"""
UNIFIED INGESTION PIPELINE FOR MIXED DOCUMENTS

This module provides a clean, optimized pipeline for ingesting "mixed" documents 
(research papers, clinical reports) into both vector and graph databases.

Key Features:
- Real PDF content extraction using pymupdf4llm and pdfplumber
- Intelligent content separation for vector vs graph databases
- Optimized for medical/research documents 
- Low-latency ingestion with parallel processing
- Production-ready database connection interfaces
"""

import re
import json
import pymupdf4llm
import pdfplumber
import pandas as pd
from pathlib import Path
# ...existing code...
from typing import Dict, List, Tuple, Any, Optional
from utility_functions import UtilityFunctions
from modelclass import ExtractedContent
# =============================================================================
from structured_document import StructuredDocumentIngestor
from unstructured_document import UnstructuredDocumentIngestor
# DATA STRUCTURES
# =============================================================================

# =============================================================================
# PDF CONTENT EXTRACTION (Based on test1.py logic)
# =============================================================================
class MixedDocumentIngestor:
    def __init__(self):
        self.structured_ingestor = StructuredDocumentIngestor()
        self.unstructured_ingestor = UnstructuredDocumentIngestor()

    def extract_mix_pdf_content(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract comprehensive content from PDF using production-ready libraries
        Based on the proven logic from test1.py with pymupdf4llm + pdfplumber
        """
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        print(f"🔍 Extracting content from: {Path(file_path).name}")
        
        extracted = {
            'text': '',
            'tables': [],
            'word_count': 0,
            'table_count': 0
        }
        
        try:
            # 1. Extract structured text with formatting preserved
            print("   📝 Extracting text with pymupdf4llm...")
            if content:
                markdown_text = content
            else:
                markdown_text = pymupdf4llm.to_markdown(file_path)
            extracted['text'] = markdown_text
            extracted['word_count'] = len(markdown_text.split())
            
            # 2. Extract tables with pdfplumber for accuracy
            print("   📊 Extracting tables with pdfplumber...")
            tables = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()         ### Kulbir needs to validate, table count is coming less
                    
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Must have header + data ## check it this line is the culprit
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                df = df.fillna('')
                                
                                # Convert to structured text
                                table_text = self.structured_ingestor.format_table_for_ingestion(df, page_num, table_idx)
                                tables.append({
                                    'page': page_num,
                                    'index': table_idx,
                                    'content': table_text,
                                    'rows': len(df),
                                    'columns': len(df.columns)
                                })
                            except Exception as e:
                                print(f"   ⚠️  Warning: Failed to process table on page {page_num}: {e}")
            
            extracted['tables'] = tables
            extracted['table_count'] = len(tables)
            
            print(f"   ✅ Extraction complete: {extracted['word_count']} words, {extracted['table_count']} tables")
            return extracted
            
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            raise



    # =============================================================================
    # CONTENT PARSING FOR DATABASE INGESTION
    # =============================================================================

    def process_mixed_document(self, file_path: str, content: Optional[str] = None) -> ExtractedContent:
        """
        Parse mixed document and separate content for vector vs graph databases.
        If content is provided, use it directly; otherwise, extract from file_path.
        """
        
        raw_content = self.extract_mix_pdf_content(file_path, content)
        full_text = raw_content['text']
        tables = raw_content['tables']
        word_count = raw_content['word_count']
        table_count = raw_content['table_count']


        # Parse narrative sections for vector database
        unstructured_chunks = self.unstructured_ingestor.ingest_unstructured_document(file_path, "mixed", full_text)        

        # Parse structured elements for graph database
        structured_data = self.structured_ingestor.extract_structured_for_graph_db(full_text, tables)

        # Extract entities and relationships
        entities = self.structured_ingestor.extract_entities(full_text)
        relationships = self.structured_ingestor.extract_relationships(full_text, entities)

        # Create metadata
        metadata = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "word_count": word_count,
            "table_count": table_count,
            "classification": "mixed",
            "document_type": UtilityFunctions.determine_document_type(full_text),
            "processing_timestamp": "2025-07-29T00:00:00Z"
        }

        return ExtractedContent(
            full_text=full_text,
            unstructured_chunks=unstructured_chunks,
            structured_data=structured_data,
            entities=entities,
            relationships=relationships,
            metadata=metadata
        )


    # =============================================================================
    # UNIFIED INGESTION PIPELINE
    # =============================================================================

    def ingest_mixed_document(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """
        Test method to analyze parse_mixed_document output only
        Database ingestion logic commented out for testing
        Accepts either file_path or content (if content is provided, it is used directly).
        """
    
        try:
            # Step 1: Parse document content and analyze output
            print("🔍 STEP 1: Parsing document content...")
            # Use provided content for parsing
            document_content = self.process_mixed_document(file_path, content=content)

            print("\n📋 DETAILED PARSING RESULTS:")
            print("=" * 50)
            # ...existing code for printing and summary...
            # print("📄 METADATA:")
            # for key, value in document_content.metadata.items():
            #     print(f"   {key}: {value}")
            # print(f"\n� NARRATIVE CHUNKS ({len(document_content.unstructured_chunks)} total):")
            # for i, chunk in enumerate(document_content.unstructured_chunks[:3]):
            #     print(f"   Chunk {i+1}: {chunk[:100]}...")
            # if len(document_content.unstructured_chunks) > 3:
            #     print(f"   ... and {len(document_content.unstructured_chunks) - 3} more chunks")
            # print(f"\n📊 STRUCTURED DATA:")
            # for key, value in document_content.structured_data.items():
            #     if isinstance(value, dict):
            #         print(f"   {key}: {len(value)} items")
            #         for subkey, subvalue in value.items():
            #             if isinstance(subvalue, list):
            #                 print(f"      {subkey}: {len(subvalue)} values - {subvalue}")
            #             else:
            #                 print(f"      {subkey}: {subvalue}")
            #     elif isinstance(value, list):
            #         print(f"   {key}: {len(value)} items - {value}")
            #     else:
            #         print(f"   {key}: {value}")
            # print(f"\n🏷️  ENTITIES ({len(document_content.entities)} total):")
            # for entity in document_content.entities[:10]:
            #     print(f"   {entity['type']}/{entity['category']}: {entity['value']}")
            # if len(document_content.entities) > 10:
            #     print(f"   ... and {len(document_content.entities) - 10} more entities")
            # print(f"\n🔗 RELATIONSHIPS ({len(document_content.relationships)} total):")
            # for rel in document_content.relationships[:5]:
            #     print(f"   {rel['source']} --{rel['relationship']}--> {rel['target']} (value: {rel.get('value', 'N/A')})")
            # if len(document_content.relationships) > 5:
            #     print(f"   ... and {len(document_content.relationships) - 5} more relationships")
            # print(f"\n📖 FULL TEXT SAMPLE ({len(document_content.full_text)} characters total):")
            # print(f"   {document_content.full_text[:500]}...")
            summary = {
                "file_path": file_path,
                "file_name": document_content.metadata["file_name"],
                "document_type": document_content.metadata["document_type"],
                "word_count": document_content.metadata["word_count"],
                "table_count": document_content.metadata["table_count"],
                "unstructured_chunks": len(document_content.unstructured_chunks),
                "entities_created": len(document_content.entities),
                "relationships_created": len(document_content.relationships),
                "metrics_extracted": sum(len(v) for v in document_content.structured_data["performance_metrics"].values()),
                "citations_found": len(document_content.structured_data["citations"]),
                "algorithms_found": len(document_content.structured_data["research_metadata"]["algorithms"]),
                "status": "parsing_complete"
            }
            print("\n" + "=" * 50)
            print("✅ PARSING ANALYSIS COMPLETE")
            print("=" * 50)
            return summary
        except Exception as e:
            print(f"\n❌ PARSING FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
        


    # class QdrantVectorDB:
    #     """Production-ready Qdrant interface for semantic search"""
    #     
    #     def __init__(self, host: str = "localhost", port: int = 6333, api_key: str = None):
    #         self.host = host
    #         self.port = port
    #         self.api_key = api_key
    #         self.collection_name = "medical_documents"
    #         # TODO: from qdrant_client import QdrantClient
    #         # self.client = QdrantClient(host=host, port=port, api_key=api_key)
    #         
    #     def ingest_narrative_chunks(self, content: ExtractedContent, batch_size: int = 100):
    #         """Ingest narrative chunks with optimized batching"""
    #         chunks = content.narrative_chunks
    #         metadata = content.metadata
    #         
    #         print(f"📊 Ingesting {len(chunks)} narrative chunks to Qdrant...")
    #         
    #         for i in range(0, len(chunks), batch_size):
    #             batch = chunks[i:i + batch_size]
    #             print(f"   Batch {i//batch_size + 1}: Processing {len(batch)} chunks")
    #             
    #             # TODO: Generate embeddings and upsert
    #             # embeddings = generate_embeddings(batch)
    #             # points = create_qdrant_points(batch, embeddings, metadata)
    #             # self.client.upsert(collection_name=self.collection_name, points=points)
    #         
    #         print(f"✅ Vector ingestion complete: {len(chunks)} chunks")

    # class Neo4jGraphDB:
    #     """Production-ready Neo4j interface for structured relationships"""
    #     
    #     def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
    #         self.uri = uri
    #         self.user = user
    #         self.password = password
    #         # TODO: from neo4j import GraphDatabase
    #         # self.driver = GraphDatabase.driver(uri, auth=(user, password))
    #         
    #     def ingest_structured_data(self, content: ExtractedContent):
    #         """Ingest structured data with optimized graph queries"""
    #         metadata = content.metadata
    #         structured = content.structured_data
    #         entities = content.entities
    #         relationships = content.relationships
    #         
    #         print(f"🕸️  Ingesting structured data to Neo4j for: {metadata['file_name']}")
    #         
    #         # Create document node
    #         print("   Creating document node...")
    #         # TODO: Execute Cypher queries
    #         
    #         # Create metric nodes
    #         metrics = structured.get("performance_metrics", {})
    #         print(f"   Creating {sum(len(v) for v in metrics.values())} metric nodes...")
    #         
    #         # Create entity nodes  
    #         print(f"   Creating {len(entities)} entity nodes...")
    #         
    #         # Create relationships
    #         print(f"   Creating {len(relationships)} relationships...")
    #         
    #         print(f"✅ Graph ingestion complete for {metadata['file_name']}")
    #         
    #     def close(self):
    #         """Close database connection"""
    #         # TODO: self.driver.close()
    #         print("🔒 Neo4j connection closed")    

    # =============================================================================
    # TEST RUNNER
    # =============================================================================

    # def test_unified_pipeline():
    #     """Test the parse_mixed_document method output only"""
        
    #     pdf_file = Path("docs/ARXIV_V5_CHESTXRAY.pdf")
        
    #     if not pdf_file.exists():
    #         print("❌ Test PDF not found!")
    #         print("Available PDF files:")
    #         for pdf in Path(".").glob("*.pdf"):
    #             print(f"   - {pdf.name}")
    #         return
        
    #     print("🧪 TESTING PARSE_MIXED_DOCUMENT OUTPUT")
    #     print("=" * 60)
        
    #     # ====== DATABASE CLIENT INITIALIZATION COMMENTED OUT ======
    #     # qdrant_client = QdrantVectorDB()
    #     # neo4j_client = Neo4jGraphDB()
        
    #     # Run parsing only (no database ingestion)
    #     result = ingest_mixed_document(str(pdf_file))
        
    #     # Display final summary
    #     print("\n📋 FINAL PARSING SUMMARY:")
    #     print("-" * 30)
    #     for key, value in result.items():
    #         print(f"   {key}: {value}")
        
    #     # ====== DATABASE CLEANUP COMMENTED OUT ======
    #     # neo4j_client.close()
        
    #     print("\n🎯 CONTENT ANALYSIS INSIGHTS:")
    #     print("   📊 Narrative chunks: Ready for vector database (semantic search)")
    #     print("   🕸️  Structured data: Ready for graph database (relationships)")
    #     print("   🏷️  Entities: Medical concepts and methodologies identified")
    #     print("   🔗 Relationships: Performance connections between algorithms and metrics")
        
    #     print("\n⚙️  PARSING OPTIMIZATIONS:")
    #     print("   • Real PDF extraction with pymupdf4llm + pdfplumber")
    #     print("   • Intelligent content separation for optimal database allocation")
    #     print("   • Section-based chunking with semantic context preservation")
    #     print("   • Medical domain entity recognition and relationship extraction")

    # if __name__ == "__main__":
    #     test_unified_pipeline()
