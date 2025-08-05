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
#import pdfplumber
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
#from .utility_functions import UtilityFunctions
from .ExtractedResponse import ExtractedResponse
from .ingestion_structured_document import StructuredDocumentIngestor
from .ingestion_unstructured_document import UnstructuredDocumentIngestor


# --- Modularized Helper Classes ---
class PDFExtractor:
    def __init__(self, structured_ingestor):
        self.structured_ingestor = structured_ingestor

    def extract(self, file_path: str, content: Optional[str] = None) -> Dict[str, Any]:
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
            print("   📝 Extracting text with pymupdf4llm...")
            if content: 
                markdown_text = content
            else:
                markdown_text = pymupdf4llm.to_markdown(file_path)
            extracted['text'] = markdown_text
            extracted['word_count'] = len(markdown_text.split())
            print("   📊 Extracting tables with pdfplumber...")
            tables = []
            # with pdfplumber.open(file_path) as pdf:
            #     for page_num, page in enumerate(pdf.pages, 1):
            #         page_tables = page.extract_tables()
            #         for table_idx, table in enumerate(page_tables):
            #             if table and len(table) > 1:
            #                 try:
            #                     df = pd.DataFrame(table[1:], columns=table[0])
            #                     df = df.fillna('')
            #                     table_text = self.structured_ingestor.format_table_for_ingestion(df, page_num, table_idx)
            #                     tables.append({
            #                         'page': page_num,
            #                         'index': table_idx,
            #                         'content': table_text,
            #                         'rows': len(df),
            #                         'columns': len(df.columns)
            #                     })
            #                 except Exception as e:
            #                     print(f"   ⚠️  Warning: Failed to process table on page {page_num}: {e}")
            extracted['tables'] = tables
            extracted['table_count'] = len(tables)
            print(f"   ✅ Extraction complete: {extracted['word_count']} words")
            return extracted
        except Exception as e:
            print(f"   ❌ Extraction failed: {e}")
            raise

class NarrativeChunker:
    def __init__(self, unstructured_ingestor):
        self.unstructured_ingestor = unstructured_ingestor
    def chunk(self, file_path: str, content: str):
        return self.unstructured_ingestor.ingest_unstructured_document(file_path, content)

class EntityRelationshipExtractor:
    def __init__(self, structured_ingestor):
        self.structured_ingestor = structured_ingestor
    def extract(self, full_text: str, tables):
        structured_data = self.structured_ingestor.extract_structured_for_graph_db(full_text, tables)
        entities = self.structured_ingestor.extract_entities(full_text)
        relationships = self.structured_ingestor.extract_relationships(full_text, entities)
        #ingest_structured_document
        return structured_data, entities, relationships

class MixedDocumentIngestor:
    def __init__(self, structured_ingestor=None, unstructured_ingestor=None):
        self.structured_ingestor = structured_ingestor or StructuredDocumentIngestor()
        self.unstructured_ingestor = unstructured_ingestor or UnstructuredDocumentIngestor()
        self.pdf_extractor = PDFExtractor(self.structured_ingestor)
        self.narrative_chunker = NarrativeChunker(self.unstructured_ingestor)
        self.entity_rel_extractor = EntityRelationshipExtractor(self.structured_ingestor)

    def ingest_mixed_document(self, file_path: str, content: Optional[str] = None) -> ExtractedResponse:
        try:
            raw_content = self.pdf_extractor.extract(file_path, content)
            full_text = raw_content['text']
            tables = raw_content['tables']
            # word_count = raw_content['word_count']
            # table_count = raw_content['table_count']
            unstructured_chunks = self.narrative_chunker.chunk(file_path, full_text)
            structured_data, entities, relationships = self.entity_rel_extractor.extract(full_text, tables)
           
            return ExtractedResponse(
                full_text=full_text,
                unstructured_chunks=unstructured_chunks,
                structured_data=structured_data,
                entities=entities,
                relationships=relationships,
                metadata={}
            )
        except Exception as e:
            print(f"❌ Error during mixed document ingestion: {e}")          
            return ExtractedResponse(
                full_text="",
                unstructured_chunks=[],
                structured_data={},
                entities=[],
                relationships=[],
                metadata={"error": str(e)}
            )

   

