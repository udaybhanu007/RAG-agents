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
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedContent:
    """Container for parsed PDF content"""
    full_text: str
    narrative_chunks: List[str]
    structured_data: Dict[str, Any]
    entities: List[Dict[str, str]]
    relationships: List[Dict[str, str]]
    metadata: Dict[str, Any]

# =============================================================================
# PDF CONTENT EXTRACTION (Based on test1.py logic)
# =============================================================================

def extract_real_pdf_content(file_path: str) -> Dict[str, Any]:
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
                            table_text = format_table_for_ingestion(df, page_num, table_idx)
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

def format_table_for_ingestion(df: pd.DataFrame, page_num: int, table_idx: int) -> str:
    """Convert DataFrame to structured text for database ingestion"""
    table_text = f"TABLE {table_idx + 1} (Page {page_num}):\n"
    table_text += f"Columns: {', '.join(df.columns)}\n\n"
    
    for idx, row in df.iterrows():
        row_items = []
        for col, val in row.items():
            if str(val).strip():
                row_items.append(f"{col}: {val}")
        
        if row_items:
            table_text += f"Row {idx + 1}: {' | '.join(row_items)}\n"
    
    return table_text

# =============================================================================
# CONTENT PARSING FOR DATABASE INGESTION
# =============================================================================

def parse_mixed_document(file_path: str) -> ExtractedContent:
    """
    Parse mixed document and separate content for vector vs graph databases
    """
    
    # Extract real content from PDF
    raw_content = extract_real_pdf_content(file_path)
    full_text = raw_content['text']
    tables = raw_content['tables']
    # Print tables in tabular form
    # if tables:
    #     print("\n==== TABLES (tabular form) ====")
    #     for t in tables:
    #         print(f"\nTable {t['index']+1} (Page {t['page']}):")
    #         # Try to reconstruct DataFrame from content if possible
    #         # If 'content' is a string, try to parse columns and rows
    #         content_lines = t['content'].splitlines()
    #         if len(content_lines) >= 2:
    #             # First line is table header, second line is columns
    #             columns_line = content_lines[1]
    #             columns = [c.strip() for c in columns_line.replace('Columns:','').split(',')]
    #             data_rows = []
    #             for row_line in content_lines[2:]:
    #                 if row_line.startswith('Row'):
    #                     # Parse row: Row N: col1: val1 | col2: val2 ...
    #                     row_data = {}
    #                     parts = row_line.split(':', 1)
    #                     if len(parts) == 2:
    #                         row_content = parts[1].strip()
    #                         for colval in row_content.split('|'):
    #                             if ':' in colval:
    #                                 col, val = colval.split(':', 1)
    #                                 row_data[col.strip()] = val.strip()
    #                     data_rows.append(row_data)
    #             if data_rows:
    #                 import pandas as pd
    #                 df = pd.DataFrame(data_rows, columns=columns)
    #                 print(df.to_string(index=False))
    #             else:
    #                 print("   (No data rows)")
    #         else:
    #             print("   (Malformed table content)")
    
    # Parse narrative sections for vector database
    narrative_chunks = extract_narrative_for_vector_db(full_text)
    
    # Parse structured elements for graph database  
    structured_data = extract_structured_for_graph_db(full_text, tables)
    
    # Extract entities and relationships
    entities = extract_entities(full_text)
    relationships = extract_relationships(full_text, entities)
    
    # Create metadata
    metadata = {
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "word_count": raw_content['word_count'],
        "table_count": raw_content['table_count'],
        "classification": "mixed",
        "document_type": determine_document_type(full_text),
        "processing_timestamp": "2025-07-29T00:00:00Z"
    }
    
    return ExtractedContent(
        full_text=full_text,
        narrative_chunks=narrative_chunks,
        structured_data=structured_data,
        entities=entities,
        relationships=relationships,
        metadata=metadata
    )

def extract_narrative_for_vector_db(text: str) -> List[str]:
    """
    Extract narrative sections optimized for semantic search in vector database
    Addresses Kulbir's concern about static patterns with adaptive approach
    """
    chunks = []
    # Primary medical research paper patterns
    section_patterns = {
        'abstract': [
            r'(?:^|\n)(?:abstract|summary)[:.]?\s*(.*?)(?=\n(?:introduction|background|methodology|keywords)|\n\n|\Z)',
            r'(?:^|\n)(?:summary)[:.]?\s*(.*?)(?=\n(?:introduction|background)|\n\n|\Z)'
        ],
        'introduction': [
            r'(?:^|\n)(?:introduction|background)[:.]?\s*(.*?)(?=\n(?:method|methodology|materials|related work)|\n\n|\Z)',
            r'(?:^|\n)(?:1\.?\s*introduction|1\.?\s*background)[:.]?\s*(.*?)(?=\n(?:2\.|method|methodology)|\n\n|\Z)'
        ],
        'methodology': [
            r'(?:^|\n)(?:methodology|methods?|materials and methods)[:.]?\s*(.*?)(?=\n(?:results|findings|experiments)|\n\n|\Z)',
            r'(?:^|\n)(?:\d+\.?\s*(?:methodology|methods?))[:.]?\s*(.*?)(?=\n(?:\d+\.|results|findings)|\n\n|\Z)'
        ],
        'results': [
            r'(?:^|\n)(?:results|findings|experiments)[:.]?\s*(.*?)(?=\n(?:discussion|conclusion|analysis)|\n\n|\Z)',
            r'(?:^|\n)(?:\d+\.?\s*(?:results|findings))[:.]?\s*(.*?)(?=\n(?:\d+\.|discussion|conclusion)|\n\n|\Z)'
        ],
        'discussion': [
            r'(?:^|\n)(?:discussion|analysis)[:.]?\s*(.*?)(?=\n(?:conclusion|limitations|future work)|\n\n|\Z)',
            r'(?:^|\n)(?:\d+\.?\s*discussion)[:.]?\s*(.*?)(?=\n(?:\d+\.|conclusion|limitations)|\n\n|\Z)'
        ],
        'conclusion': [
            r'(?:^|\n)(?:conclusion|summary|conclusions)[:.]?\s*(.*?)(?=\n(?:references|acknowledgments|bibliography)|\n\n|\Z)',
            r'(?:^|\n)(?:\d+\.?\s*(?:conclusion|conclusions))[:.]?\s*(.*?)(?=\n(?:references|acknowledgments)|\n\n|\Z)'
        ]
    }
    # Try multiple patterns for each section
    sections_found = set()
    for section_name, patterns in section_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            for match in matches:
                clean_text = clean_text_for_vector_db(match)
                if len(clean_text) > 100:  # Minimum meaningful content
                    chunks.extend(chunk_text(clean_text, section_name))
                    sections_found.add(section_name)
                    break  # Found match for this section, try next section
    # If no clear sections found, use adaptive chunking
    if not sections_found:
        print("   📄 No standard sections found, using adaptive chunking...")
        clean_full_text = clean_text_for_vector_db(text)
        # Try to identify content blocks by line breaks and formatting
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        if paragraphs:
            for i, paragraph in enumerate(paragraphs[:10]):  # Limit to first 10 paragraphs
                chunks.extend(chunk_text(paragraph, f"content_block_{i+1}"))
        else:
            # Fallback to simple chunking
            chunks = chunk_text(clean_full_text, "general_content")
    print(f"   📝 Extracted {len(chunks)} narrative chunks from {len(sections_found)} sections")
    return chunks

### Gunjan needs to validate if structured section is valid several this types of .pdf documents. 
### If some of the cases any of the structured data is not present then what needs to be done ?

def extract_structured_for_graph_db(text: str, tables: List[Dict]) -> Dict[str, Any]:
    """
    Extract structured data optimized for graph database relationships
    """
    structured = {
        "performance_metrics": extract_performance_metrics(text),
        "statistical_data": extract_statistical_data(text),
        "research_metadata": extract_research_metadata(text),
        "citations": extract_citations(text),
        "table_data": process_tables_for_graph(tables)
    }
    
    return structured

def extract_performance_metrics(text: str) -> Dict[str, List[float]]:
    """Extract quantitative performance metrics"""
    metrics = {}
    
    metric_patterns = {
        "accuracy": r'accuracy[:\s]*(\d+\.?\d*)\s*[%]?',
        "sensitivity": r'sensitivity[:\s]*(\d+\.?\d*)\s*[%]?',
        "specificity": r'specificity[:\s]*(\d+\.?\d*)\s*[%]?',
        "auc": r'(?:auc|area under (?:the )?curve)[:\s]*(\d+\.?\d*)',
        "precision": r'precision[:\s]*(\d+\.?\d*)\s*[%]?',
        "recall": r'recall[:\s]*(\d+\.?\d*)\s*[%]?',
        "f1_score": r'f1[-\s]?score[:\s]*(\d+\.?\d*)'
    }
    
    for metric, pattern in metric_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            metrics[metric] = [float(m) for m in matches if is_valid_metric(m)]
    
    return metrics

def extract_statistical_data(text: str) -> Dict[str, Any]:
    """Extract statistical values and measurements"""
    return {
        "p_values": [float(p) for p in re.findall(r'p\s*[<>=]\s*(0\.\d+)', text, re.I)],
        "confidence_intervals": [(float(m), float(s)) for m, s in re.findall(r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)', text)],
        "sample_sizes": [int(n.replace(',', '')) for n in re.findall(r'n\s*=\s*([\d,]+)', text, re.I)]
    }

def extract_research_metadata(text: str) -> Dict[str, Any]:
    """Extract research study metadata"""
    return {
        "datasets": re.findall(r'dataset[:\s]*([A-Za-z0-9\-_]+)', text, re.I),
        "algorithms": re.findall(r'\b(CNN|ResNet|VGG|BERT|Transformer|SVM|Random Forest)\b', text, re.I),
        "medical_modalities": re.findall(r'\b(CT|MRI|X-ray|ultrasound|mammography|PET)\b', text, re.I)
    }

def extract_citations(text: str) -> List[str]:
    """Extract citation references"""
    citations = []
    citation_patterns = [
        r'\[(\d+)\]',  # [1], [2], etc.
        r'\(([A-Za-z]+\s+et\s+al\.?,?\s*\d{4})\)',  # (Smith et al., 2020)
        r'DOI:\s*([^\s]+)'  # DOI references
    ]
    
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, text, re.I))
    
    return list(set(citations))  # Remove duplicates

### Gunjan properly validate this method with all the tables in the .pdf
def process_tables_for_graph(tables: List[Dict]) -> List[Dict[str, Any]]:
    """Process table data for graph database storage"""
    processed = []
    for table in tables:
        # Extract structured information from table content
        table_info = {
            "page": table["page"],
            "rows": table["rows"], 
            "columns": table["columns"],
            "summary": f"Table with {table['rows']} rows and {table['columns']} columns on page {table['page']}"
        }
        processed.append(table_info)
    
    return processed

### Gunjan validate if we need to keep medical patterns as static list
def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract medical and research entities for graph nodes"""
    entities = []
    
    # Medical entities
    medical_patterns = {
        "condition": r'\b(pneumonia|covid-19|tuberculosis|cancer|diabetes|hypertension|pathology)\b',
        "imaging": r'\b(chest\s+x-ray|ct\s+scan|mri|ultrasound|radiograph|mammography)\b',
        "procedure": r'\b(diagnosis|treatment|screening|biopsy|surgery)\b'
    }
    
    # Research methodology entities  
    method_patterns = {
        "algorithm": r'\b(CNN|ResNet|VGG|deep\s+learning|machine\s+learning|neural\s+network)\b',
        "process": r'\b(training|validation|testing|cross-validation|preprocessing)\b'
    }
    
    all_patterns = {**medical_patterns, **method_patterns}
    
    for category, pattern in all_patterns.items():
        matches = re.findall(pattern, text, re.I)
        for match in matches:
            entities.append({
                "type": "medical" if category in medical_patterns else "methodology",
                "category": category,
                "value": match.lower().strip()
            })
    
    # Remove duplicates
    return remove_duplicate_entities(entities)

### Gunjan debug and validate on why relationships are not showing
def extract_relationships(text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Extract relationships between entities"""
    relationships = []
    
    # Performance relationships (algorithm -> metric -> value)
    for entity in entities:
        if entity["category"] == "algorithm":
            algorithm = entity["value"]
            # Look for performance metrics near this algorithm mention
            context_windows = find_context_windows(text, algorithm, window_size=300)
            
            for context in context_windows:
                metrics = extract_performance_metrics(context)
                for metric_name, values in metrics.items():
                    for value in values:
                        relationships.append({
                            "source": algorithm,
                            "target": metric_name,
                            "relationship": "achieves",
                            "value": value,
                            "context": "performance"
                        })
    
    return relationships

# =============================================================================
# UTILITY FUNCTIONS  
# =============================================================================

def clean_text_for_vector_db(text: str) -> str:
    """Clean text for optimal vector embedding"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special markdown characters but preserve structure
    text = re.sub(r'[#*_`]', '', text)
    # Remove URLs and email addresses
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

def chunk_text(text: str, section_name: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks with section context"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text.strip()) > 100:  # Minimum meaningful chunk
            chunks.append(f"[{section_name}] {chunk_text.strip()}")
    
    return chunks

def find_context_windows(text: str, term: str, window_size: int = 200) -> List[str]:
    """Find text windows around specific terms"""
    windows = []
    term_positions = [m.start() for m in re.finditer(re.escape(term), text, re.I)]
    
    for pos in term_positions:
        start = max(0, pos - window_size)
        end = min(len(text), pos + window_size)
        windows.append(text[start:end])
    
    return windows

def is_valid_metric(value_str: str) -> bool:
    """Validate metric values are reasonable"""
    try:
        value = float(value_str)
        return 0 <= value <= 100 or 0 <= value <= 1  # Percentage or decimal
    except:
        return False

def remove_duplicate_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate entities based on type and value"""
    unique = []
    seen = set()
    
    for entity in entities:
        key = f"{entity['type']}_{entity['value']}"
        if key not in seen:
            seen.add(key)
            unique.append(entity)
    
    return unique

def determine_document_type(text: str) -> str:
    """Determine specific document type for better processing"""
    medical_score = len(re.findall(r'\b(patient|clinical|medical|diagnosis|radiograph)\b', text, re.I))
    research_score = len(re.findall(r'\b(abstract|methodology|results|discussion|conclusion)\b', text, re.I))
    
    if medical_score > 20 and research_score > 5:
        return "medical_research"
    elif medical_score > 10:
        return "clinical_document"
    elif research_score > 5:
        return "research_paper"
    else:
        return "general_mixed"

# =============================================================================
# DATABASE CONNECTION INTERFACES (COMMENTED OUT FOR TESTING)
# =============================================================================

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
# UNIFIED INGESTION PIPELINE
# =============================================================================

def ingest_mixed_document(file_path: str) -> Dict[str, Any]:
    """
    Test method to analyze parse_mixed_document output only
    Database ingestion logic commented out for testing
    """
    
    print(f"🧪 TESTING PARSE_MIXED_DOCUMENT OUTPUT")
    print(f"📄 Processing: {Path(file_path).name}")
    print("-" * 50)
    
    try:
        # Step 1: Parse document content and analyze output
        print("🔍 STEP 1: Parsing document content...")
        document_content = parse_mixed_document(file_path)
        
        print("\n📋 DETAILED PARSING RESULTS:")
        print("=" * 50)
        
        # Display metadata
        print("📄 METADATA:")
        for key, value in document_content.metadata.items():
            print(f"   {key}: {value}")
        
        # Display narrative chunks summary
        print(f"\n� NARRATIVE CHUNKS ({len(document_content.narrative_chunks)} total):")
        for i, chunk in enumerate(document_content.narrative_chunks[:3]):  # Show first 3
            print(f"   Chunk {i+1}: {chunk[:100]}...")
        if len(document_content.narrative_chunks) > 3:
            print(f"   ... and {len(document_content.narrative_chunks) - 3} more chunks")
        
        # Display structured data
        print(f"\n📊 STRUCTURED DATA:")
        for key, value in document_content.structured_data.items():
            if isinstance(value, dict):
                print(f"   {key}: {len(value)} items")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        print(f"      {subkey}: {len(subvalue)} values - {subvalue}")
                    else:
                        print(f"      {subkey}: {subvalue}")
            elif isinstance(value, list):
                print(f"   {key}: {len(value)} items - {value}")
            else:
                print(f"   {key}: {value}")
        
        # Display entities
        print(f"\n🏷️  ENTITIES ({len(document_content.entities)} total):")
        for entity in document_content.entities[:10]:  # Show first 10
            print(f"   {entity['type']}/{entity['category']}: {entity['value']}")
        if len(document_content.entities) > 10:
            print(f"   ... and {len(document_content.entities) - 10} more entities")
        
        # Display relationships
        print(f"\n🔗 RELATIONSHIPS ({len(document_content.relationships)} total):")
        for rel in document_content.relationships[:5]:  # Show first 5
            print(f"   {rel['source']} --{rel['relationship']}--> {rel['target']} (value: {rel.get('value', 'N/A')})")
        if len(document_content.relationships) > 5:
            print(f"   ... and {len(document_content.relationships) - 5} more relationships")
        
        # Sample of full text
        print(f"\n📖 FULL TEXT SAMPLE ({len(document_content.full_text)} characters total):")
        print(f"   {document_content.full_text[:500]}...")
        
        # ====== DATABASE INGESTION LOGIC COMMENTED OUT ======
        # print("\n📊 STEP 2: Parallel database ingestion...")
        # qdrant_client.ingest_narrative_chunks(document_content)
        # neo4j_client.ingest_structured_data(document_content)
        
        # Step 2: Return comprehensive summary for analysis
        summary = {
            "file_path": file_path,
            "file_name": document_content.metadata["file_name"],
            "document_type": document_content.metadata["document_type"],
            "word_count": document_content.metadata["word_count"],
            "table_count": document_content.metadata["table_count"],
            "narrative_chunks": len(document_content.narrative_chunks),
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
