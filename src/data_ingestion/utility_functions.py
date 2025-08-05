import re
from typing import Dict, List

class UtilityFunctions:

    @staticmethod
    def create_qdrant_points(texts, embeddings, metadatas, ids=None):
        """
        Create Qdrant points for a batch of texts, embeddings, and metadatas.
        Each point includes a unique id (from ids if provided, else uuid4), the embedding vector, and the payload (metadata + chunk).
        """
        import uuid
        points = []
        if ids is None:
            ids = [None] * len(texts)
        for text, embedding, metadata, point_id in zip(texts, embeddings, metadatas, ids):
            combined_metadata = {**metadata, "chunk": text}
            if point_id is None:
                point_id = str(uuid.uuid4())
            point = {
                "id": point_id,
                "vector": embedding,
                "payload": combined_metadata,
            }
            points.append(point)
        return points
    
    @staticmethod
    def clean_text_for_vector_db(text: str) -> str:
        """Clean text for optimal vector embedding"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[#*_`]', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        return text.strip()

    # @staticmethod
    # def chunk_text(text: str, section_name: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    #     words = text.split()
    #     chunks = []
    #     for i in range(0, len(words), chunk_size - overlap):
    #         chunk_words = words[i:i + chunk_size]
    #         chunk_text = ' '.join(chunk_words)
    #         if len(chunk_text.strip()) > 100:
    #             chunks.append(f"[{section_name}] {chunk_text.strip()}")
    #     return chunks

    @staticmethod
    def find_context_windows(text: str, term: str, window_size: int = 200) -> List[str]:
        windows = []
        term_positions = [m.start() for m in re.finditer(re.escape(term), text, re.I)]
        for pos in term_positions:
            start = max(0, pos - window_size)
            end = min(len(text), pos + window_size)
            windows.append(text[start:end])
        return windows

    @staticmethod
    def is_valid_metric(value_str: str) -> bool:
        try:
            value = float(value_str)
            return 0 <= value <= 100 or 0 <= value <= 1
        except:
            return False

    @staticmethod
    def remove_duplicate_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        unique = []
        seen = set()
        for entity in entities:
            key = f"{entity['type']}_{entity['value']}"
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique

    @staticmethod
    def determine_document_type(text: str) -> str:
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
    
    # @staticmethod
    # def save_extracted_content(folder: str, content: str, output_file: str):
    #     """
    #     Save all extracted content to a markdown file in the extracted_content folder.
    #     Always postfix 'extracted_unstructured.md' to the output_file name.
    #     """
    #     import os
    #     from pathlib import Path

    #     #folder = "extracted_content"
    #     os.makedirs(folder, exist_ok=True)

    #     # Remove any path from output_file, just use the base name
    #     base_name = os.path.basename(output_file)
    #     file_path = os.path.join(folder, base_name + "_extracted_unstructured.md")

    #     print(f"Saving extracted content to {file_path}...")
    #     with open(file_path, 'w', encoding='utf-8') as f:
    #         f.write("# PDF Content Extraction Report\n\n")
    #         f.write("## Document Text\n\n")
    #         f.write(content)
    #         f.write("\n\n")
    #     print(f"✓ Content saved to {file_path}")

   
    
    @staticmethod
    def contains_citation(text: str) -> bool:
        """Check if the text contains typical citation patterns."""
        if not text:
            return False
        # Common patterns: [1], (Smith et al., 2020), [12,13], (2020), etc.
        citation_patterns = [
            r'\[\d+(,\s*\d+)*\]',  # [1] or [1, 2]
            r'\([A-Za-z][^\)]*et al\.,? \d{4}\)',  # (Smith et al., 2020)
            r'\([A-Za-z][^\)]*, \d{4}\)',  # (Smith, 2020)
            r'\(\d{4}\)'  # (2020)
        ]
        return any(re.search(p, text) for p in citation_patterns)

    @staticmethod
    def remove_citations(text: str) -> str:
        """Remove common citation patterns from the text."""
        if not text:
            return text
        # Remove [1], [1,2], [12, 13]
        text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
        # Remove (Smith et al., 2020)
        text = re.sub(r'\([A-Za-z][^\)]*et al\.,? \d{4}\)', '', text)
        # Remove (Smith, 2020)
        text = re.sub(r'\([A-Za-z][^\)]*, \d{4}\)', '', text)
        # Remove (2020)
        text = re.sub(r'\(\d{4}\)', '', text)
        # Remove extra spaces left by removals
        text = re.sub(r'\s{2,}', ' ', text)
        return text