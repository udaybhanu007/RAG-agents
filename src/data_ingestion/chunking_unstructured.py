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

def chunk_with_overlap(paragraphs, chunk_size=3, overlap=1):
    """Create overlapping chunks for better context."""
    if not paragraphs:
        return []
    
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(paragraphs), step):
        chunk = paragraphs[i:i + chunk_size]
        if chunk:
            chunks.append("\n\n".join(chunk))
    
    return chunks

def measure_chunk_precision(chunks):
    # Simple precision: average words per chunk, chunk count, min/max/avg length
    if not chunks:
        return {"num_chunks": 0, "min_words": 0, "max_words": 0, "avg_words": 0}
    
    lengths = [len(chunk.split()) for chunk in chunks]
    return {
        "num_chunks": len(chunks),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "avg_words": sum(lengths) / len(lengths)
    }

def split_large_chunks(chunks, max_words=400):
    """Split chunks that exceed the maximum word limit."""
    refined_chunks = []
    
    for chunk in chunks:
        word_count = len(chunk.split())
        
        if word_count <= max_words:
            refined_chunks.append(chunk)
        else:
            # Split by sentences for better semantic boundaries
            sentences = re.split(r'(?<=[.!?])\s+', chunk)
            current_chunk = []
            current_words = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                
                if current_words + sentence_words > max_words and current_chunk:
                    refined_chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_words = sentence_words
                else:
                    current_chunk.append(sentence)
                    current_words += sentence_words
            
            if current_chunk:
                refined_chunks.append(" ".join(current_chunk))
    
    return refined_chunks

def create_chunk(file_path: str, content: str, chunk_size=3, overlap=1, min_words=30, max_words=400):
    """
    Create paragraph-based overlapping chunks from markdown content.
    Args:
        file_path (str): Path to the file.
        chunk_size (int): Number of paragraphs per chunk.
        overlap (int): Number of overlapping paragraphs between chunks.
        min_words (int): Minimum words in a paragraph before merging.
        max_words (int): Maximum words per chunk before splitting.
    Returns:
        List[str]: List of chunked text blocks.
    """
    
    paragraphs = split_markdown_paragraphs(content)
    merged_paragraphs = merge_short_paragraphs(paragraphs, min_words)
    chunks = chunk_with_overlap(merged_paragraphs, chunk_size, overlap)
    
    # Split oversized chunks
    refined_chunks = split_large_chunks(chunks, max_words)
    
    # Print chunk statistics
    if refined_chunks:
        lengths = [len(chunk.split()) for chunk in refined_chunks]
        print(f"   📊 Chunk Stats - Count: {len(refined_chunks)}, "
              f"Avg Words: {sum(lengths)/len(lengths):.1f}, "
              f"Range: {min(lengths)}-{max(lengths)} words")
    
    return refined_chunks

