from utility_functions import UtilityFunctions
from typing import List, Dict, Any
import re
from mixed_document import ExtractedContent
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


class UnstructuredDocumentIngestor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant_client = self._setup_qdrant()
        self.collection_name = "medical_research_doc"

    def _setup_qdrant(self) -> QdrantClient:
        """Setup Qdrant client"""
        return QdrantClient(
            url="https://f779f36d-3ee0-4afe-b35c-9ced9a62f083.us-west-1-0.aws.cloud.qdrant.io:6333",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.VmFb90WfFOv7XrIfi2YNmtA8gHhCdWq34gLE2jxYCIM"
        )

    def ingest_unstructured_document(self, file_path: str, classification: str, content: str = None) -> dict:
        """
        Ingest an unstructured document.
        """
        from pathlib import Path
        folder = "extracted_content"
        UtilityFunctions.save_extracted_content(folder, content, file_path)
        chunks = self.extract_chunk_for_vector_db(file_path, content)
        # Create metadata
        metadata = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "word_count": len(content.split()),
            "classification": classification,
            "document_type": UtilityFunctions.determine_document_type(content),
            "processing_timestamp": "2025-07-29T00:00:00Z"
        }

        # For unstructured documents, structured_data can be an empty dict
        structured_data = {}
        unstructured = ExtractedContent(
            full_text=content,
            unstructured_chunks=chunks,
            structured_data=structured_data,
            entities=[],
            relationships=[],
            metadata=metadata
        )

        self.ingest_narrative_chunks(unstructured)

        print(f"[Structured Ingestion] Processing: {file_path}")
        return {
            "file_path": file_path,
            "status": "structured_ingestion_complete",
            "content_sample": content[:500] if content else None
        }

    @staticmethod
    def get_section_patterns() -> dict:
        """
        Returns regex patterns for common research paper sections.
        """
        return {
            'abstract': [
                r'(?:^|\n)(?:abstract|summary)[:.]?\s*(.*?)(?=\n(?:introduction|background|methodology|keywords)|\n\n|\Z)',
                r'(?:^|\n)(?:summary)[:.]?\s*(.*?)(?=\n(?:introduction|background)|\n\n|\Z)'
            ],
            'introduction': [
                r'(?:^|\n)(?:introduction|background)[:.]?\s*(.*?)(?=\n(?:method|methodology|materials|related work)|\n\n|\Z)',
                r'(?:^|\n)(?:1\.?\s*introduction|1\.?\s*background)[:.]?\s*(.*?)(?=\n(?:2\.|method|methodology)|\n\n|\Z)'
            ],
            'methodology': [
                r'(?:^|\n)(?:methodology|methods?|materials and methods)[:.]?\s*(.*?)(?=\n(?:results|findings|experiments)|\n\n|\Z)',
                r'(?:^|\n)(?:\d+\.?\s*(?:methodology|methods?))[:.]?\s*(.*?)(?=\n(?:\d+\.|results|findings)|\n\n|\Z)'
            ],
            'results': [
                r'(?:^|\n)(?:results|findings|experiments)[:.]?\s*(.*?)(?=\n(?:discussion|conclusion|analysis)|\n\n|\Z)',
                r'(?:^|\n)(?:\d+\.?\s*(?:results|findings))[:.]?\s*(.*?)(?=\n(?:\d+\.|discussion|conclusion)|\n\n|\Z)'
            ],
            'discussion': [
                r'(?:^|\n)(?:discussion|analysis)[:.]?\s*(.*?)(?=\n(?:conclusion|limitations|future work)|\n\n|\Z)',
                r'(?:^|\n)(?:\d+\.?\s*discussion)[:.]?\s*(.*?)(?=\n(?:\d+\.|conclusion|limitations)|\n\n|\Z)'
            ],
            'conclusion': [
                r'(?:^|\n)(?:conclusion|summary|conclusions)[:.]?\s*(.*?)(?=\n(?:references|acknowledgments|bibliography)|\n\n|\Z)',
                r'(?:^|\n)(?:\d+\.?\s*(?:conclusion|conclusions))[:.]?\s*(.*?)(?=\n(?:references|acknowledgments)|\n\n|\Z)'
            ]
        }

    def extract_chunk_for_vector_db(self, file_path: str, content: str) -> List[str]:
        """
        Extract narrative sections optimized for semantic search in vector database.
        """
        text = content
        chunks: List[str] = []
        section_patterns = self.get_section_patterns()

        sections_found = set()
        for section_name, patterns in section_patterns.items():
            found = False
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                for match in matches:
                    clean_text = UtilityFunctions.clean_text_for_vector_db(match)
                    if len(clean_text) > 100:
                        chunked = UtilityFunctions.chunk_text(clean_text, section_name)
                        chunks.extend(chunked)
                        sections_found.add(section_name)
                        found = True
                        break
                if found:
                    break

        if not sections_found:
            print("   📄 No standard sections found, using adaptive chunking...")
            clean_full_text = UtilityFunctions.clean_text_for_vector_db(text)
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
            if paragraphs:
                for i, paragraph in enumerate(paragraphs[:10]):
                    chunked = UtilityFunctions.chunk_text(paragraph, f"content_block_{i+1}")
                    chunks.extend(chunked)
            else:
                chunks = UtilityFunctions.chunk_text(clean_full_text, "general_content")

        print(f"   📝 Extracted {len(chunks)} narrative chunks from {len(sections_found)} sections")
        return chunks

    def ingest_narrative_chunks(self, content: ExtractedContent, batch_size: int = 100):
        """Ingest narrative chunks with optimized batching and embedding generation."""
        chunks = content.unstructured_chunks
        metadata = content.metadata

        print(f"📊 Ingesting {len(chunks)} narrative chunks to Qdrant...")
        self.create_collection()
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"   Batch {i//batch_size + 1}: Processing {len(batch)} chunks")
            # update metadata word_count
            metadata["word_count"] = sum(len(chunk.split()) for chunk in batch)

            # Generate embeddings for the batch
            embeddings = self.embedding_model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            # Create Qdrant points
            points = UtilityFunctions.create_qdrant_points(batch, embeddings, metadata)

            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)

        print(f"✅ Vector ingestion complete: {len(chunks)} chunks")

    

    def create_collection(self):
        """Ensure Qdrant collection and payload index exist, and clear data if collection exists."""
        from qdrant_client.http.exceptions import UnexpectedResponse
        try:
            # Check if collection exists
            collection_exists = False
            try:
                self.qdrant_client.get_collection(self.collection_name)
                collection_exists = True
            except Exception:
                collection_exists = False

            if collection_exists:
                # Delete the collection entirely
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                print(f"✓ Deleted existing collection: {self.collection_name}")

            # Always create the collection (either it didn't exist, or we just deleted it)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"✓ Created medical collection: {self.collection_name}")

            # # Always ensure payload index exists for quality_score
            # try:
            #     self.qdrant_client.create_payload_index(
            #         collection_name=self.collection_name,
            #         field_name="quality_score",
            #         type="float"
            #     )
            #     print(f"✓ Ensured payload index for 'quality_score' exists.")
            # except UnexpectedResponse as e:
            #     if "already exists" in str(e):
            #         print(f"Payload index for 'quality_score' already exists.")
            #     else:
            #         print(f"Error creating payload index: {e}")
        except Exception as e:
            print(f"Collection error: {e}")