"""
Generic Document Classification System

This module provides a unified approach to classify documents in a knowledge repository as:
- "structured": Data-heavy documents (CSV, Excel, XML, JSON, or documents with primarily tables)
- "unstructured": Text-heavy documents (plain text, or documents with primarily narrative content)  
- "mixed": Documents with both structured data and narrative text (research papers, reports)

Key Features:
- Single method (classify_document) handles all file types including PDFs
- Content-aware analysis for ambiguous formats like PDF/Word docs
- Optimized for medical/research documents with specialized pattern detection
- Batch processing for entire directories
"""

import re
import os
import pymupdf4llm
from pathlib import Path
from unified_ingestion_pipeline import ingest_mixed_document
# File extension constants for document classification
STRUCTURED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".xml", ".json", ".yaml", ".yml"]
UNSTRUCTURED_EXTENSIONS = [".txt"]
MIXED_POTENTIAL_EXTENSIONS = [".pdf", ".doc", ".docx", ".rtf", ".odt"]

def get_file_extension(file_path):
    """Extract file extension from file path"""
    return Path(file_path).suffix.lower()

def extract_content_sample(file_path, max_chars=5000):
    """Extract sample content from file for analysis"""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        
        extension = file_path.suffix.lower()
        
        if extension == ".txt":
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(max_chars)
        elif extension == ".pdf":
            markdown_text = pymupdf4llm.to_markdown(file_path)
            return markdown_text        
        elif extension in [".doc", ".docx"]:
            return f"Word document: {file_path.name} - Content extraction would require python-docx library"
        else:
            # For other text-based files, try to read as text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(max_chars)
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"

def analyze_content_structure(content):
    """
    Analyze document content to determine structure type
    Works for PDFs, Word docs, and any text-based content
    
    Returns:
    - "structured": Primarily tables, data, forms
    - "unstructured": Primarily narrative text
    - "mixed": Combination of structured data and narrative text
    """
    has_tables = has_significant_tables_or_data(content)
    has_narrative = has_significant_narrative_text(content)
    
    # Check for research paper indicators
    research_indicators = [
        len(re.findall(r'(abstract|introduction|methodology|results|discussion|conclusion)', content, re.I)) >= 3,
        len(re.findall(r'(references|bibliography|citations)', content, re.I)) > 0,
        len(re.findall(r'\[\d+\]|\(\d{4}\)', content)) > 5,  # Citation patterns
        len(re.findall(r'(doi|pmid|issn)', content, re.I)) > 0,  # Academic identifiers
    ]
    
    # Check for medical/clinical indicators
    medical_indicators = [
        len(re.findall(r'(chest x-ray|radiograph|ct scan|mri|ultrasound)', content, re.I)) > 0,
        len(re.findall(r'(pneumonia|diagnosis|treatment|patient|clinical)', content, re.I)) > 5,
        len(re.findall(r'(sensitivity|specificity|accuracy|auc)', content, re.I)) > 0,
    ]
    
    is_research_paper = any(research_indicators)
    is_medical_content = any(medical_indicators)
    
    # Classification logic
    if has_tables and has_narrative:
        return "mixed"  # Documents with both data tables and narrative
    elif has_tables and not has_narrative:
        return "structured"  # Primarily data/tables
    elif has_narrative and (is_research_paper or is_medical_content):
        return "mixed"  # Research papers typically have some structured elements
    elif has_narrative:
        return "unstructured"  # Primarily narrative text
    else:
        return "unstructured"  # Default for unclear cases

def classify_document(file_path):
    """
    Generic document classification for any document type
    Returns: (classification, content)
    classification: "structured", "unstructured", or "mixed"
    content: extracted content sample (str)
    """
    extension = get_file_extension(file_path).lower()
    content = extract_content_sample(file_path, max_chars=5000)
    
    # Special healthcare format handling
    if extension in [".hl7", ".fhir", ".cda", ".ccr", ".ccd"]:
        return "mixed", content
    # Structured formats are always structured
    if extension in STRUCTURED_EXTENSIONS:
        return "structured", content
    # Document formats are always unstructured for plain text
    elif extension in UNSTRUCTURED_EXTENSIONS:
        return "unstructured", content
    # Mixed potential formats (PDFs, Word docs, etc.) need content analysis
    elif extension in MIXED_POTENTIAL_EXTENSIONS:
        return analyze_content_structure(content), content
    # Content-based classification for unknown extensions
    else:
        return analyze_content_structure(content), content

def has_significant_tables_or_data(content):
    """Check if document has substantial structured data"""
    table_indicators = [
        len(re.findall(r'\|\s*\w+\s*\|', content)) > 5,  # Table borders
        len(re.findall(r'\t\w+\t', content)) > 10,        # Tab-separated
        len(re.findall(r'\d+\.\d+\s*mg', content)) > 5,   # Medical measurements
        len(re.findall(r'Table\s+\d+[:.]', content, re.I)) > 2,  # Table references
        len(re.findall(r'Figure\s+\d+[:.]', content, re.I)) > 3,  # Figure references
        len(re.findall(r'\d+\.\d+\s*[%]', content)) > 5,  # Percentages
        len(re.findall(r'p\s*[<>=]\s*0\.\d+', content, re.I)) > 3,  # Statistical p-values
        len(re.findall(r'\d+\s*±\s*\d+', content)) > 3,   # Plus-minus statistics
    ]
    return any(table_indicators)

def has_significant_narrative_text(content):
    """Check if document contains substantial narrative/unstructured text"""
    # Enhanced patterns for medical narrative text
    narrative_patterns = [
        r'patient\s+(reports|states|complains|describes)',
        r'history\s+of\s+present\s+illness',
        r'physical\s+examination\s+reveals',
        r'assessment\s+and\s+plan',
        r'(introduction|background)[:.].*?(method|methodology)',  # Research paper sections
        r'(discussion|conclusion)[:.].*?',  # Discussion sections
        r'case\s+(study|report|presentation)',  # Case studies
        r'we\s+(observed|found|discovered|analyzed)',  # Research narrative
        r'(results|findings)\s+(showed|demonstrated|indicated)',  # Results narrative
    ]
    
    narrative_score = sum(len(re.findall(pattern, content, re.I)) for pattern in narrative_patterns)
    
    # Also check for paragraph-like structure
    paragraph_indicators = [
        len(re.findall(r'\.\s+[A-Z]', content)) > 10,  # Sentence endings followed by capitals
        len(content.split()) > 200,  # Substantial word count
        len(re.findall(r'\b(the|and|or|but|however|therefore|moreover)\b', content, re.I)) > 20  # Common connecting words
    ]
    
    return narrative_score > 2 or any(paragraph_indicators)

def analyze_repository_documents(directory_path):
    """
    Analyze all documents in the knowledge repository and classify them
    """
    results = {}
    directory = Path(directory_path)
    
    if not directory.exists():
        return {"error": f"Directory {directory_path} does not exist"}
    
    # Get all files in the directory
    for file_path in directory.iterdir():
        if file_path.is_file():
            try:
                classification = classify_document(str(file_path))
                file_size = file_path.stat().st_size
                
                results[file_path.name] = {
                    "classification": classification,
                    "extension": file_path.suffix.lower(),
                    "size_bytes": file_size,
                    "path": str(file_path)
                }
            except Exception as e:
                results[file_path.name] = {
                    "classification": "error",
                    "error": str(e),
                    "path": str(file_path)
                }
    
    return results

# def generate_classification_summary(results):
#     """
#     Generate a summary of classification results
#     """
#     if "error" in results:
#         return results
    
#     summary = {
#         "total_files": len(results),
#         "by_classification": {},
#         "by_extension": {},
#         "files_by_type": {
#             "structured": [],
#             "unstructured": [],
#             "mixed": [],
#             "error": []
#         }
#     }
    
#     for filename, info in results.items():
#         classification = info.get("classification", "error")
#         extension = info.get("extension", "unknown")
        
#         # Count by classification
#         summary["by_classification"][classification] = summary["by_classification"].get(classification, 0) + 1
        
#         # Count by extension
#         summary["by_extension"][extension] = summary["by_extension"].get(extension, 0) + 1
        
#         # Group files by type
#         summary["files_by_type"][classification].append({
#             "filename": filename,
#             "extension": extension,
#             "size_bytes": info.get("size_bytes", 0)
#         })
    
#     return summary

# Test the classification system
if __name__ == "__main__":

    directory_path = "docs"
    results = analyze_repository_documents(directory_path)
    #loop results
    for filename, info in results.items():
        print(f"File: {filename}")
        if not isinstance(info, dict):
            print(f"  Error: {info}")
            continue
        classification = info['classification']
        if isinstance(classification, tuple):
            classification, _ = classification
        print(f"  Classification: {classification}")
        print(f"  Size (bytes): {info['size_bytes'] if 'size_bytes' in info else 0}")
        print(f"  Path: {info['path'] if 'path' in info else ''}")
        print()

        if classification == 'mixed':
            result =ingest_mixed_document(info['path'])
            print(f"   ✅ Ingested mixed document: {info['path']}")
            # Display final summary
            print("\n📋 FINAL PARSING SUMMARY:")
            print("-" * 30)
            for key, value in result.items():
                print(f"   {key}: {value}")
        elif classification == 'structured':
            print(f"   ✅ Structured document detected: {info['path']}")
        elif classification == 'unstructured':
            print(f"   ✅ Unstructured document detected: {info['path']}")

   
    
    print("\n=== Test Complete ===")