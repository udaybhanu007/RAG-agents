from .classify_document import classify_document, analyze_repository_documents
from .mixed_document import MixedDocumentIngestor
from .ingestion_structured_document import StructuredDocumentIngestor
from .ingestion_unstructured_document import UnstructuredDocumentIngestor

class DocumentIngestionOrchestrator:
    """
    Adapter class to unify ingestion for mixed, structured, and unstructured documents.
    """
    def __init__(self, mixed_ingestor=None, structured_ingestor=None, unstructured_ingestor=None):
        self.mixed_ingestor = mixed_ingestor or MixedDocumentIngestor()
        self.structured_ingestor = structured_ingestor or StructuredDocumentIngestor()
        self.unstructured_ingestor = unstructured_ingestor or UnstructuredDocumentIngestor()

    def process_document(self, file_path: str) -> dict:
        """
        Classifies the document and ingests it based on its type.
        """
        doc_type, content = classify_document(file_path)
        print(f"Document type for {file_path}: {doc_type}")
        if doc_type == "mixed":
            ingestion_result = self.mixed_ingestor.ingest_mixed_document(file_path, content)
        elif doc_type == "structured":
            ingestion_result = self.structured_ingestor.ingest_structured_document(file_path, content) # type: ignore
        elif doc_type == "unstructured":
            ingestion_result = self.unstructured_ingestor.ingest_unstructured_document(file_path, content) # type: ignore
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
        return {"classification": doc_type, "content": content, "ingestion_result": ingestion_result}

    def analyze_directory(self, directory_path: str) -> dict:
        """
        Analyzes all documents in the given directory and classifies them.       
        """
        return analyze_repository_documents(directory_path)


def ingest_directory(adapter: DocumentIngestionOrchestrator, directory: str):
    """
    Classifies and ingests all files in the specified directory using the adapter.
    Prints results for each file.
    """
    dir_results = adapter.analyze_directory(directory)
    print("Directory classification results:")
    for filename, info in dir_results.items():
        if not isinstance(info, dict) or "path" not in info:
            print(f"Skipping {filename}: {info}")
            continue
        file_path = info["path"]
        print(f"\nIngesting {filename} ({file_path})...")
        try:
            ingest_result = adapter.process_document(file_path)
            print(f"Ingestion successfully done for {filename}")
        except Exception as e:
            print(f"Error ingesting {filename}: {e}")


# if __name__ == "__main__":
#     adapter = DocumentIngestionAdapter()     
#     ingest_directory(adapter, "docs")
