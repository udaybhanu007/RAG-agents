import re
import os
from pathlib import Path

def split_markdown_paragraphs(md_text):
    # Split by double newlines, skip page headers
    paragraphs = []
    for para in md_text.split('\n\n'):
        para = para.strip()
        if para.startswith('## Page '):
            continue
        if para:
            paragraphs.append(para)
    return paragraphs

def merge_short_paragraphs(paragraphs, min_words=30):
    merged = []
    buffer = ""
    for para in paragraphs:
        # Always treat page headers as their own paragraph
        if para.startswith('## Page '):
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(para)
        elif len(para.split()) < min_words:
            buffer += " " + para
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(para)
    if buffer:
        merged.append(buffer.strip())
    return merged

def chunk_with_overlap(paragraphs, chunk_size=2, overlap=1):
    # Each chunk is a single paragraph, no overlap
    return [[p] for p in paragraphs]

def measure_chunk_precision(chunks):
    # Simple precision: average words per chunk, chunk count, min/max/avg length
    lengths = [len(chunk.split()) for chunk in chunks]
    return {
        "num_chunks": len(chunks),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "avg_words": sum(lengths) / len(lengths)
    }

def create_chunk(file_path: str, content: str, chunk_size=2, overlap=1, min_words=30):
    """
    Create paragraph-based overlapping chunks from markdown content.
    Args:
        file_path (str, optional): Path to the file.
        chunk_size (int): Number of paragraphs per chunk.
        overlap (int): Number of overlapping paragraphs between chunks.
        min_words (int): Minimum words in a paragraph before merging.
    Returns:
        List[str]: List of chunked text blocks.
    """
    
    paragraphs = split_markdown_paragraphs(content)
    merged_paragraphs = merge_short_paragraphs(paragraphs, min_words=30)
    chunks = chunk_with_overlap(merged_paragraphs, chunk_size=1, overlap=0)
    #precision = measure_chunk_precision(chunks)
    return chunks

