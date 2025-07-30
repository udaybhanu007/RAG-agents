import re
from typing import Dict, List

class UtilityFunctions:
    @staticmethod
    def clean_text_for_vector_db(text: str) -> str:
        """Clean text for optimal vector embedding"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[#*_`]', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        return text.strip()

    @staticmethod
    def chunk_text(text: str, section_name: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            if len(chunk_text.strip()) > 100:
                chunks.append(f"[{section_name}] {chunk_text.strip()}")
        return chunks

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
