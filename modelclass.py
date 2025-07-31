from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ExtractedContent:
    """Container for parsed document content"""
    full_text: str
    unstructured_chunks: List[str]
    structured_data: Dict[str, Any]
    entities: List[Dict[str, str]]
    relationships: List[Dict[str, str]]
    metadata: Dict[str, Any]
