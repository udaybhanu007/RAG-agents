import re
import json
import pymupdf4llm
import pdfplumber
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from utility_functions import UtilityFunctions

class StructuredDocumentIngestor:
    def __init__(self):
        pass

    def ingest_structured_document(self, file_path: str, content: str = None) -> dict:
        """
        Ingest a structured document (CSV, Excel, XML, JSON, etc.).
        This is a placeholder implementation. Replace with actual logic as needed.

        Args:
            file_path (str): Path to the structured document file.
            content (str, optional): Extracted content of the file, if available.

        Returns:
            dict: Ingestion result.
        """
        print(f"[Structured Ingestion] Processing: {file_path}")
        # Example: parse CSV, Excel, or JSON here
        return {
            "file_path": file_path,
            "status": "structured_ingestion_complete",
            "content_sample": content[:500] if content else None
        }


    ### Gunjan properly validate this method with all the tables in the .pdf
    def process_tables_for_graph(self, tables: List[Dict]) -> List[Dict[str, Any]]:
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
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
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
        return UtilityFunctions.remove_duplicate_entities(entities)

    ### Gunjan debug and validate on why relationships are not showing
    def extract_relationships(self, text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Performance relationships (algorithm -> metric -> value)
        for entity in entities:
            if entity["category"] == "algorithm":
                algorithm = entity["value"]
                # Look for performance metrics near this algorithm mention
                context_windows = UtilityFunctions.find_context_windows(text, algorithm, window_size=300)
                for context in context_windows:
                    metrics = self.extract_performance_metrics(context)
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

    def extract_performance_metrics(self, text: str) -> Dict[str, List[float]]:
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
                metrics[metric] = [float(m) for m in matches if UtilityFunctions.is_valid_metric(m)]
        
        return metrics

    def extract_statistical_data(self, text: str) -> Dict[str, Any]:
        """Extract statistical values and measurements"""
        return {
            "p_values": [float(p) for p in re.findall(r'p\s*[<>=]\s*(0\.\d+)', text, re.I)],
            "confidence_intervals": [(float(m), float(s)) for m, s in re.findall(r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)', text)],
            "sample_sizes": [int(n.replace(',', '')) for n in re.findall(r'n\s*=\s*([\d,]+)', text, re.I)]
        }

    def extract_research_metadata(self, text: str) -> Dict[str, Any]:
        """Extract research study metadata"""
        return {
            "datasets": re.findall(r'dataset[:\s]*([A-Za-z0-9\-_]+)', text, re.I),
            "algorithms": re.findall(r'\b(CNN|ResNet|VGG|BERT|Transformer|SVM|Random Forest)\b', text, re.I),
            "medical_modalities": re.findall(r'\b(CT|MRI|X-ray|ultrasound|mammography|PET)\b', text, re.I)
        }

    def extract_citations(self, text: str) -> List[str]:
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


    def format_table_for_ingestion(self, df: pd.DataFrame, page_num: int, table_idx: int) -> str:
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

    def extract_structured_for_graph_db(self, text: str, tables: List[Dict]) -> Dict[str, Any]:
        """
        Extract structured data optimized for graph database relationships
        """
        structured = {
            "performance_metrics": self.extract_performance_metrics(text),
            "statistical_data": self.extract_statistical_data(text),
            "research_metadata": self.extract_research_metadata(text),
            "citations": self.extract_citations(text),
            "table_data": self.process_tables_for_graph(tables)
        }
        return structured