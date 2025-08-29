"""
Document chunking utilities for Graphiti ingestion
Simplified semantic chunking following rate limiting guidelines
"""

import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentChunker:
    """Simplified document chunker for Graphiti with semantic awareness"""
    
    def __init__(self, max_chunk_size: int = 12000):
        """
        Initialize document chunker with larger, semantic chunks
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters (default: 12KB for better semantic coherence)
        """
        self.max_chunk_size = max_chunk_size
    
    def create_semantic_chunks(self, text: str, document_name: str = "Unknown") -> List[str]:
        """
        Create fewer, larger, semantically meaningful chunks
        Following guidelines for rate limiting optimization
        
        Args:
            text: Input text to chunk
            document_name: Name of the document for logging
            
        Returns:
            List of chunk strings
        """
        if not text or len(text) <= self.max_chunk_size:
            logger.info(f"Document '{document_name}' doesn't need chunking (length: {len(text)})")
            return [text]
        
        # Split by document structure rather than arbitrary length
        sections = self._split_by_sections(text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for section in sections:
            if current_size + len(section) < self.max_chunk_size:
                current_chunk += section
                current_size += len(section)
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section
                current_size = len(section)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Document '{document_name}' split into {len(chunks)} semantic chunks")
        return chunks
    
    def _split_by_sections(self, content: str) -> List[str]:
        """
        Split by document structure rather than arbitrary length
        Optimized for medical/research documents
        """
        # Common section markers for research documents
        section_markers = [
            "Background & Motivation:",
            "Details:",
            "Contents:",
            "Limitations:",
            "Acknowledgement:",
            "Reference:",
            "Abstract:",
            "Introduction:",
            "Methods:",
            "Results:",
            "Discussion:",
            "Conclusion:",
            "co-occurrence matrix"
        ]
        
        sections = []
        current_section = ""
        
        for line in content.split('\n'):
            if any(marker in line for marker in section_markers):
                if current_section:
                    sections.append(current_section)
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section:
            sections.append(current_section)
            
        return sections
