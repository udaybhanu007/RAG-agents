
COLLECTION_NAME = "my_medical_research_doc"


from typing import List, Dict, Any
from mixed_document import ExtractedContent
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from chunk_markdown import create_chunk
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class QdrantDBManager:
    """Manages Qdrant client and collection operations."""
    def __init__(self, api_url: str, api_key: str, collection_name: str, vector_size: int = 384):
        self.client = QdrantClient(url=api_url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._collection_initialized = False

    def create_collection(self):
        """Ensure Qdrant collection exists; do not delete if already present."""
        try:
            try:
                self.client.get_collection(self.collection_name)
                print(f"✓ Collection already exists: {self.collection_name}")
                return
            except Exception:
                pass
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            print(f"✓ Created collection: {self.collection_name}")
            self._collection_initialized = True
        except Exception as e:
            print(f"Collection error: {e}")

    def upsert_data(self, points: list):
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            print(f"Error upserting points: {e}")

class DocumentChunker:
    """Chunks documents into smaller parts."""
    def extract_chunks(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        from utility_functions import UtilityFunctions
        chunk_list = []
        print("   📄 Chunking by paragraph...")
        md_text = content       

        try:
            # create_chunk can take md_text directly if refactored, else write to temp file
            chunks = create_chunk(file_path, md_text)
        except ImportError as e:
            print(f"Error importing chunk_markdown.create_chunk: {e}")
            chunks = []
        for idx, chunk in enumerate(chunks):
            # If chunk is a list (from chunk_markdown), join it to a string
            if isinstance(chunk, list):
                chunk_str = "\n\n".join(chunk)
            else:
                chunk_str = chunk
            from datetime import datetime
            chunk_id = idx + 1
            chunk_metadata = {
                "file_path": file_path,
                "chunk_id": chunk_id,
                "chunk_word_count": len(chunk_str.split()),
                "created_date": datetime.now().strftime("%Y-%m-%d"),
            }
            chunk_list.append({"chunk": chunk_str, "metadata": chunk_metadata})
        print(f"   📝 Extracted {len(chunk_list)} narrative chunks using paragraph-based chunking.")
        return chunk_list

class EmbeddingManager:
    """Manages embedding generation using SentenceTransformer."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    def generate_embeddings(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

class UnstructuredDocumentIngestor:
    def __init__(self, api_url=None, api_key=None):
        # Load from environment or .env file
        self.collection_name = COLLECTION_NAME
        self.api_url = api_url or os.environ.get("QDRANT_API_URL")
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        if not self.api_url or not self.api_key:
            raise ValueError("QDRANT_API_URL and QDRANT_API_KEY must be set in the environment or .env file.")
        self.chunker = DocumentChunker()
        self._collection_initialized = False

    def ingest_unstructured_document(self, file_path: str, content : str, classification: str = "un-structured") -> 'ExtractedContent': # type: ignore
        # Ensure Qdrant collection exists before chunking/ingestion
        self.qdrant_manager = QdrantDBManager(self.api_url, self.api_key, self.collection_name)  # type: ignore
        self.embedding_manager = EmbeddingManager()      
        chunk_list = self.chunker.extract_chunks(file_path, content)
        # For now, full_text is empty, but you can extract and pass the actual text if needed
        unstructured = ExtractedContent(
            full_text="",
            unstructured_chunks=chunk_list, # type: ignore
            structured_data={},
            entities=[],
            relationships=[],
            metadata={}
        )
        self.ingest_narrative_chunks(unstructured)
        print(f"[Unstructured Ingestion] Processing: {file_path}")
        return unstructured

    def ingest_narrative_chunks(self, content: ExtractedContent, batch_size: int = 100):
        from utility_functions import UtilityFunctions
        import hashlib
        chunk_dicts = content.unstructured_chunks
        print(f"📊 Ingesting {len(chunk_dicts)} narrative chunks to Qdrant...")
        if not self._collection_initialized:
            self.qdrant_manager.create_collection()
            self._collection_initialized = True
        # Use file_hash and chunk_id from each chunk's metadata for unique IDs
        for i in range(0, len(chunk_dicts), batch_size):
            batch = chunk_dicts[i:i + batch_size]
            print(f"   Batch {i // batch_size + 1}: Processing {len(batch)} chunks")
            batch_chunks = []
            batch_metadatas = []
            for item in batch:
                chunk_text = item.get("chunk", item) if isinstance(item, dict) else item
                # Remove citations if present
                if UtilityFunctions.contains_citation(chunk_text):
                    chunk_text = UtilityFunctions.remove_citations(chunk_text)
                # Clean text for vector DB
                chunk_text = UtilityFunctions.clean_text_for_vector_db(chunk_text)
                batch_chunks.append(chunk_text)
                batch_metadatas.append(item.get("metadata", {}) if isinstance(item, dict) else {})
            embeddings = self.embedding_manager.generate_embeddings(batch_chunks)
            # Use utility function to create Qdrant points
            points = UtilityFunctions.create_qdrant_points(batch_chunks, embeddings, batch_metadatas)           
            try:
                self.qdrant_manager.upsert_data(points)           
            except Exception as e:
                print(f"[ERROR] Failed to upsert points to Qdrant: {e}")
        print(f"✅ Vector ingestion complete: {len(chunk_dicts)} chunks")