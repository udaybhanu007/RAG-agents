
# --- Imports ---
import json
import re
import time
import pandas as pd
import spacy
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# --- Data structure for parsed document content ---
@dataclass
class ExtractedContent:
    full_text: str
    narrative_chunks: List[str]
    structured_data: Dict[str, Any]
    entities: List[Dict[str, str]]
    relationships: List[Dict[str, str]]
    metadata: Dict[str, Any]

def format_table_for_ingestion(df: pd.DataFrame, page_num: int, table_idx: int) -> str:
    """
    Format a DataFrame as a readable table string for logging or output.
    """
    table_text = f"TABLE {table_idx + 1} (Page {page_num}):\n"
    table_text += f"Columns: {', '.join(df.columns)}\n\n"
    for idx, row in df.iterrows():
        row_items = [f"{col}: {val}" for col, val in row.items() if str(val).strip()]
        if row_items:
            table_text += f"Row {idx + 1}: {' | '.join(row_items)}\n"
    return table_text

def determine_document_type(text: str) -> str:
    """
    Heuristically classify the document type based on keyword frequency.
    """
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

def extract_entities_static(text: str) -> List[Dict[str, str]]:
    """
    Extract static (regex-based) medical and methodology entities from text.
    """
    entities = []
    patterns = {
        # Medical concepts
        "condition": r'\b(pneumonia|covid-19|tuberculosis|cancer|diabetes|hypertension|pathology)\b',
        "imaging": r'\b(chest\s+x-ray|ct\s+scan|mri|ultrasound|radiograph|mammography)\b',
        "procedure": r'\b(diagnosis|treatment|screening|biopsy|surgery)\b',
        # Methodology
        "algorithm": r'\b(CNN|ResNet|VGG|deep\s+learning|machine\s+learning|neural\s+network)\b',
        "process": r'\b(training|validation|testing|cross-validation|preprocessing)\b'
    }
    for category, pattern in patterns.items():
        matches = re.findall(pattern, text, re.I)
        for match in matches:
            entities.append({
                "type": "medical" if category in ["condition", "imaging", "procedure"] else "methodology",
                "category": category,
                "value": match.lower().strip()
            })
    return remove_duplicate_entities(entities)

def remove_duplicate_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove duplicate entities by type and value.
    """
    unique = []
    seen = set()
    for entity in entities:
        key = f"{entity['type']}_{entity['value']}"
        if key not in seen:
            seen.add(key)
            unique.append(entity)
    return unique

def extract_relationships_static(text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extract static relationships between algorithms and performance metrics.
    """
    relationships = []
    for entity in entities:
        if entity["category"] == "algorithm":
            algorithm = entity["value"]
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

def find_context_windows(text: str, term: str, window_size: int = 200) -> List[str]:
    """
    Find text windows around a term for context-based extraction.
    """
    windows = []
    term_positions = [m.start() for m in re.finditer(re.escape(term), text, re.I)]
    for pos in term_positions:
        start = max(0, pos - window_size)
        end = min(len(text), pos + window_size)
        windows.append(text[start:end])
    return windows

def parse_mixed_document(file_path: str) -> ExtractedContent:
    ext = Path(file_path).suffix.lower()
    # Use extract_content for all file types
    text, tables = extract_content(Path(file_path))
    # For PDFs, also extract formatted tables
    formatted_tables = []
    if ext == '.pdf' and tables:
        for idx, tbl in enumerate(tables):
            try:
                df = pd.DataFrame(tbl["content"])
                formatted_tables.append(format_table_for_ingestion(df, tbl.get("page", 1), idx))
            except Exception:
                pass
    # Narrative chunks
    narrative_chunks = extract_narrative_for_vector_db(text)
    # Structured data
    structured_data = extract_structured_for_graph_db(text, tables)
    # Entities (dynamic + static)
    entities = extract_entities(text, tables) + extract_entities_static(text)
    # Relationships (dynamic + static)
    relationships = extract_relationships(text, tables) + extract_relationships_static(text, entities)
    # Metadata
    metadata = {
        "file_path": str(file_path),
        "file_name": Path(file_path).name,
        "word_count": len(text.split()),
        "table_count": len(tables),
        "classification": "mixed",
        "document_type": determine_document_type(text),
        "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    return ExtractedContent(
        full_text=text,
        narrative_chunks=narrative_chunks,
        structured_data=structured_data,
        entities=entities,
        relationships=relationships,
        metadata=metadata
    )

def ingest_mixed_document(file_path: str) -> Dict[str, Any]:
    print(f"🧪 TESTING PARSE_MIXED_DOCUMENT OUTPUT")
    print(f"📄 Processing: {Path(file_path).name}")
    print("-" * 50)
    try:
        print("🔍 STEP 1: Parsing document content...")
        document_content = parse_mixed_document(file_path)
        print("\n📋 DETAILED PARSING RESULTS:")
        print("=" * 50)
        print("📄 METADATA:")
        for key, value in document_content.metadata.items():
            print(f"   {key}: {value}")
        print(f"\n🏷️  ENTITIES ({len(document_content.entities)} total):")
        for entity in document_content.entities[:10]:
            print(f"   {entity['type']}/{entity['category']}: {entity['value']}")
        if len(document_content.entities) > 10:
            print(f"   ... and {len(document_content.entities) - 10} more entities")
        print(f"\n🔗 RELATIONSHIPS ({len(document_content.relationships)} total):")
        for rel in document_content.relationships[:5]:
            print(f"   {rel['source']} --{rel['relationship']}--> {rel['target']} (value: {rel.get('value', 'N/A')})")
        if len(document_content.relationships) > 5:
            print(f"   ... and {len(document_content.relationships) - 5} more relationships")
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
        print(f"\n📖 FULL TEXT SAMPLE ({len(document_content.full_text)} characters total):")
        print(f"   {document_content.full_text[:500]}...")
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

def test_unified_pipeline():
    pdf_file = Path("Docs/ARXIV_V5_CHESTXRAY.pdf")
    if not pdf_file.exists():
        print("❌ Test PDF not found!")
        print("Available PDF files:")
        for pdf in Path(".").glob("*.pdf"):
            print(f"   - {pdf.name}")
        return
    print("🧪 TESTING PARSE_MIXED_DOCUMENT OUTPUT")
    print("=" * 60)
    result = ingest_mixed_document(str(pdf_file))
    print("\n📋 FINAL PARSING SUMMARY:")
    print("-" * 30)
    for key, value in result.items():
        print(f"   {key}: {value}")
    print("\n🎯 CONTENT ANALYSIS INSIGHTS:")
    print("   📊 Narrative chunks: Ready for vector database (semantic search)")
    print("   🕸️  Structured data: Ready for graph database (relationships)")
    print("   🏷️  Entities: Medical concepts and methodologies identified")
    print("   🔗 Relationships: Performance connections between algorithms and metrics")
    print("\n⚙️  PARSING OPTIMIZATIONS:")
    print("   • Real PDF extraction with pymupdf4llm + pdfplumber")
    print("   • Intelligent content separation for optimal database allocation")
    print("   • Section-based chunking with semantic context preservation")
    print("   • Medical domain entity recognition and relationship extraction")

# class QdrantVectorDB:
#     """Production-ready Qdrant interface for semantic search"""
#     def __init__(self, host: str = "localhost", port: int = 6333, api_key: str = None):
#         self.host = host
#         self.port = port
#         self.api_key = api_key
#         self.collection_name = "medical_documents"
#         # TODO: from qdrant_client import QdrantClient
#         # self.client = QdrantClient(host=host, port=port, api_key=api_key)
#     def ingest_narrative_chunks(self, content: ExtractedContent, batch_size: int = 100):
#         chunks = content.narrative_chunks
#         metadata = content.metadata
#         print(f"📊 Ingesting {len(chunks)} narrative chunks to Qdrant...")
#         for i in range(0, len(chunks), batch_size):
#             batch = chunks[i:i + batch_size]
#             print(f"   Batch {i//batch_size + 1}: Processing {len(batch)} chunks")
#             # TODO: Generate embeddings and upsert
#             # embeddings = generate_embeddings(batch)
#             # points = create_qdrant_points(batch, embeddings, metadata)
#             # self.client.upsert(collection_name=self.collection_name, points=points)
#         print(f"✅ Vector ingestion complete: {len(chunks)} chunks")

# class Neo4jGraphDB:
#     """Production-ready Neo4j interface for structured relationships"""
#     def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
#         self.uri = uri
#         self.user = user
#         self.password = password
#         # TODO: from neo4j import GraphDatabase
#         # self.driver = GraphDatabase.driver(uri, auth=(user, password))
#     def ingest_structured_data(self, content: ExtractedContent):
#         metadata = content.metadata
#         structured = content.structured_data
#         entities = content.entities
#         relationships = content.relationships
#         print(f"🕸️  Ingesting structured data to Neo4j for: {metadata['file_name']}")
#         print("   Creating document node...")
#         # TODO: Execute Cypher queries
#         metrics = structured.get("performance_metrics", {})
#         print(f"   Creating {sum(len(v) for v in metrics.values())} metric nodes...")
#         print(f"   Creating {len(entities)} entity nodes...")
#         print(f"   Creating {len(relationships)} relationships...")
#         print(f"✅ Graph ingestion complete for {metadata['file_name']}")
#     def close(self):
#         # TODO: self.driver.close()
#         print("🔒 Neo4j connection closed")

import json, re, time
import pandas as pd
import spacy
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Load spaCy with extended limit
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 30_000_000

@dataclass
class ExtractedContent:
    full_text: str
    narrative_chunks: List[str]
    structured_data: Dict[str, Any]
    entities: List[Dict[str, str]]
    relationships: List[Dict[str, str]]
    metadata: Dict[str, Any]

# -------------------------------------
# FILE PARSER (multi-format)
# -------------------------------------
def extract_content(file_path):
    """
    Extract text and tables from a file (PDF, CSV, Excel, JSON, XML, DOCX).
    Returns (text, tables) tuple.
    """
    ext = file_path.suffix.lower()
    text, tables = '', []
    try:
        if ext == '.pdf':
            import pymupdf4llm, pdfplumber
            text = pymupdf4llm.to_markdown(file_path)
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    for t in page.extract_tables():
                        if t and len(t) > 1:
                            df = pd.DataFrame(t[1:], columns=t[0]).fillna('')
                            tables.append({'page': i+1, 'content': df.to_dict(), 'rows': len(df), 'columns': len(df.columns)})
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            text = df.head(1000).to_string(index=False)
            tables.append({'content': df.to_dict(), 'rows': len(df), 'columns': len(df.columns)})
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
            text = df.head(1000).to_string(index=False)
            tables.append({'content': df.to_dict(), 'rows': len(df), 'columns': len(df.columns)})
        elif ext == '.json':
            data = json.load(open(file_path, encoding='utf-8'))
            text = json.dumps(data)[:50000]
        elif ext == '.xml':
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            text = ET.tostring(tree.getroot(), encoding='unicode')[:50000]
        elif ext == '.docx':
            import docx
            doc = docx.Document(file_path)
            text = '\n'.join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
    return text, tables

# -------------------------------------
# ENTITY EXTRACTION
# -------------------------------------
def extract_entities(text, tables):
    import string
    def is_number(val):
        try:
            float(val)
            return True
        except:
            return False

    def is_filename(val):
        return bool(re.fullmatch(r"[\w\-]+_\d+\.(png|jpg|jpeg|dcm)", val.strip(), re.IGNORECASE))

    def is_empty(val):
        return not val or str(val).strip().lower() in {"nan", "none", "null", ""}

    def is_clinical_column(col, vals):
        # Heuristic: clinical if most values are non-numeric, non-file, non-empty, and not too many unique values
        non_empty = [v for v in vals if not is_empty(v)]
        if not non_empty:
            return False
        non_numeric = [v for v in non_empty if not is_number(v)]
        non_file = [v for v in non_numeric if not is_filename(str(v))]
        unique_vals = set(str(v).strip().lower() for v in non_file if str(v).strip())
        # If >60% of non-empty values are non-numeric/non-file and unique count is not too high, treat as clinical
        ratio = len(non_file) / len(non_empty) if non_empty else 0
        return ratio > 0.6 and 1 < len(unique_vals) < 200

    entities = []
    seen = set()
    for tbl in tables:
        for col, vals in tbl.get("content", {}).items():
            col_clean = str(col).strip()
            if not is_clinical_column(col_clean, vals):
                continue
            # Add clinical column headers as entities
            if col_clean and col_clean.lower() not in seen:
                entities.append({
                    "type": "table_column",
                    "category": "column",
                    "value": col_clean
                })
                seen.add(col_clean.lower())
            # For each value in the column
            for v in vals:
                if is_empty(v) or is_number(v) or is_filename(str(v)):
                    continue
                # For columns with 'finding' or 'label', split by '|'
                if "finding" in col.lower() or "label" in col.lower():
                    for finding in str(v).split("|"):
                        finding = finding.strip()
                        if is_empty(finding) or is_number(finding) or is_filename(finding):
                            continue
                        key = ("medical_finding", finding.lower())
                        if key not in seen:
                            entities.append({
                                "type": "medical_finding",
                                "category": col_clean,
                                "value": finding
                            })
                            seen.add(key)
                else:
                    val_clean = str(v).strip()
                    key = (col_clean, val_clean.lower())
                    if val_clean and key not in seen:
                        entities.append({
                            "type": "table_value",
                            "category": col_clean,
                            "value": val_clean
                        })
                        seen.add(key)
    return entities

# -------------------------------------
# RELATIONSHIP EXTRACTION
# -------------------------------------
def extract_relationships(text, tables):
    rels = []
    # Only create relationships between image filename and finding label, and between image/finding and bounding box as a single object
    for tbl in tables:
        content = tbl.get("content", {})
        cols = list(content.keys())
        # Identify likely column names
        img_col = None
        finding_col = None
        bbox_cols = []
        for c in cols:
            cl = c.lower()
            if "image" in cl or "file" in cl or "index" in cl:
                img_col = c
            elif "finding" in cl or "label" in cl:
                finding_col = c
            elif "bbox" in cl or cl in {"x", "y", "w", "h"} or "coord" in cl:
                bbox_cols.append(c)
        row_count = min(500, len(content[cols[0]])) if cols else 0
        for i in range(row_count):
            # Get image, finding, and bbox values for this row
            img = str(content[img_col][i]).strip() if img_col and img_col in content else None
            finding = str(content[finding_col][i]).strip() if finding_col and finding_col in content else None
            bbox = None
            if bbox_cols:
                bbox_vals = [str(content[c][i]).strip() for c in bbox_cols if c in content and i < len(content[c])]
                # Only create bbox if all values are present and are numbers
                if all(bbox_vals) and all(re.fullmatch(r"-?\d+(\.\d+)?", v) for v in bbox_vals):
                    bbox = {c: float(content[c][i]) for c in bbox_cols}
            # Relationship: image -> finding
            if img and finding and img != "nan" and finding != "nan":
                rels.append({
                    "source": img,
                    "relationship": "has_finding",
                    "target": finding
                })
            # Relationship: image/finding -> bbox (as a single object)
            if img and bbox and img != "nan":
                rels.append({
                    "source": img,
                    "relationship": "has_bbox",
                    "target": bbox
                })
            if finding and bbox and finding != "nan":
                rels.append({
                    "source": finding,
                    "relationship": "has_bbox",
                    "target": bbox
                })
    return dedup(rels)

# -------------------------------------
# NARRATIVE CHUNKING & STRUCTURED DATA EXTRACTION
# -------------------------------------
def extract_narrative_for_vector_db(text: str) -> List[str]:
    chunks = []
    section_patterns = {
        'abstract': r'(?:^|\n)(?:abstract|summary)[:.]?\s*(.*?)(?=\n(?:introduction|background|methodology|keywords)|\n\n|\Z)',
        'introduction': r'(?:^|\n)(?:introduction|background)[:.]?\s*(.*?)(?=\n(?:method|methodology|materials|related work)|\n\n|\Z)',
        'methodology': r'(?:^|\n)(?:methodology|methods?|materials and methods)[:.]?\s*(.*?)(?=\n(?:results|findings|experiments)|\n\n|\Z)',
        'results': r'(?:^|\n)(?:results|findings|experiments)[:.]?\s*(.*?)(?=\n(?:discussion|conclusion|analysis)|\n\n|\Z)',
        'discussion': r'(?:^|\n)(?:discussion|analysis)[:.]?\s*(.*?)(?=\n(?:conclusion|limitations|future work)|\n\n|\Z)',
        'conclusion': r'(?:^|\n)(?:conclusion|summary|conclusions)[:.]?\s*(.*?)(?=\n(?:references|acknowledgments|bibliography)|\n\n|\Z)'
    }
    for section_name, pattern in section_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        for match in matches:
            clean_text = clean_text_for_vector_db(match)
            if len(clean_text) > 100:
                chunks.extend(chunk_text(clean_text, section_name))
    if not chunks:
        clean_full_text = clean_text_for_vector_db(text)
        chunks = chunk_text(clean_full_text, "general_content")
    return chunks

def extract_structured_for_graph_db(text: str, tables: List[Dict]) -> Dict[str, Any]:
    structured = {
        "performance_metrics": extract_performance_metrics(text),
        "statistical_data": extract_statistical_data(text),
        "research_metadata": extract_research_metadata(text),
        "citations": extract_citations(text),
        "table_data": process_tables_for_graph(tables)
    }
    return structured

def extract_performance_metrics(text: str) -> Dict[str, List[float]]:
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
    return {
        "p_values": [float(p) for p in re.findall(r'p\s*[<>=]\s*(0\.\d+)', text, re.I)],
        "confidence_intervals": [(float(m), float(s)) for m, s in re.findall(r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)', text)],
        "sample_sizes": [int(n.replace(',', '')) for n in re.findall(r'n\s*=\s*([\d,]+)', text, re.I)]
    }

def extract_research_metadata(text: str) -> Dict[str, Any]:
    return {
        "datasets": re.findall(r'dataset[:\s]*([A-Za-z0-9\-_]+)', text, re.I),
        "algorithms": re.findall(r'\b(CNN|ResNet|VGG|BERT|Transformer|SVM|Random Forest)\b', text, re.I),
        "medical_modalities": re.findall(r'\b(CT|MRI|X-ray|ultrasound|mammography|PET)\b', text, re.I)
    }

def extract_citations(text: str) -> List[str]:
    citations = []
    citation_patterns = [
        r'\[(\d+)\]',
        r'\(([A-Za-z]+\s+et\s+al\.?\,?\s*\d{4})\)',
        r'DOI:\s*([^\s]+)'
    ]
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, text, re.I))
    return list(set(citations))

def process_tables_for_graph(tables: List[Dict]) -> List[Dict[str, Any]]:
    processed = []
    for table in tables:
        table_info = {
            "rows": table.get("rows", 0),
            "columns": table.get("columns", 0),
            "summary": f"Table with {table.get('rows', 0)} rows and {table.get('columns', 0)} columns"
        }
        processed.append(table_info)
    return processed

# -------------------------------------
# DEDUPLICATION
# -------------------------------------
def dedup(items):
    """
    Remove duplicate dicts from a list (by JSON serialization).
    """
    seen, out = set(), []
    for x in items:
        j = json.dumps(x, sort_keys=True)
        if j not in seen:
            seen.add(j)
            out.append(x)
    return out

def clean_text_for_vector_db(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[#*_`]', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

def chunk_text(text: str, section_name: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        if len(chunk_text.strip()) > 100:
            chunks.append(f"[{section_name}] {chunk_text.strip()}")
    return chunks

def is_valid_metric(value_str: str) -> bool:
    try:
        value = float(value_str)
        return 0 <= value <= 100 or 0 <= value <= 1
    except:
        return False

# -------------------------------------
# MAIN FUNCTION
# -------------------------------------

def main():

    # --- Main batch processing for all supported files in source_document ---
    src = Path(__file__).parent / "source_document"
    out = Path(__file__).parent / "output_folder"
    out.mkdir(exist_ok=True)

    for f in src.glob("*"):
        if f.suffix.lower() not in ['.pdf', '.csv', '.xls', '.xlsx', '.json', '.xml', '.docx']:
            continue
        print(f"\n🔍 Processing: {f.name}")
        start = time.time()

        text, tables = extract_content(f)
        narrative_chunks = extract_narrative_for_vector_db(text)
        structured_data = extract_structured_for_graph_db(text, tables)
        ents = extract_entities(text, tables)
        rels = extract_relationships(text, tables)

        metadata = {
            "file_path": str(f),
            "file_name": f.name,
            "word_count": len(text.split()),
            "table_count": len(tables),
            "classification": "mixed",
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        # Save all outputs
        (out / f"{f.stem}_text.txt").write_text(text, encoding='utf-8')
        json.dump(ents, open(out / f"{f.stem}_entities.json", 'w', encoding='utf-8'), indent=2)
        json.dump(rels, open(out / f"{f.stem}_relationships.json", 'w', encoding='utf-8'), indent=2)
        json.dump(narrative_chunks, open(out / f"{f.stem}_narrative_chunks.json", 'w', encoding='utf-8'), indent=2)
        json.dump(structured_data, open(out / f"{f.stem}_structured_data.json", 'w', encoding='utf-8'), indent=2)
        json.dump(metadata, open(out / f"{f.stem}_metadata.json", 'w', encoding='utf-8'), indent=2)

        print(f"✅ Done in {round(time.time() - start, 2)}s | Entities: {len(ents)}, Relationships: {len(rels)}, Narrative Chunks: {len(narrative_chunks)}, Tables: {len(tables)}")

        # --- Validation: Check if output entities/relationships are present in the source file ---
        # For CSV, XLS, XLSX: check if at least some entities/relationships match table headers/values
        if f.suffix.lower() in ['.csv', '.xls', '.xlsx']:
            try:
                df = pd.read_csv(f) if f.suffix.lower() == '.csv' else pd.read_excel(f)
                headers = set(df.columns)
                values = set(str(v).strip() for v in df.values.flatten() if str(v).strip())
                entity_values = set(e['value'] for e in ents)
                # Check if at least 50% of headers are present as entity columns
                header_matches = sum(1 for h in headers if h in entity_values)
                if header_matches < max(1, len(headers)//2):
                    print(f"⚠️  Validation warning: Only {header_matches}/{len(headers)} headers found as entities.")
                # Check if at least 10 table values are present as entities
                value_matches = sum(1 for v in values if v in entity_values)
                if value_matches < 10:
                    print(f"⚠️  Validation warning: Only {value_matches} table values found as entities.")
            except Exception as e:
                print(f"⚠️  Validation error: {e}")
        # For text-based files: check if at least some entities are present in the text
        elif f.suffix.lower() in ['.pdf', '.json', '.xml', '.docx']:
            try:
                entity_values = set(e['value'] for e in ents)
                text_sample = text[:10000]
                found = sum(1 for v in entity_values if v in text_sample)
                if found < 5:
                    print(f"⚠️  Validation warning: Only {found} entities found in the first 10k chars of text.")
            except Exception as e:
                print(f"⚠️  Validation error: {e}")

if __name__ == "__main__":
    main()
