# adapter_document_ingestion.py
from classify_document import classify_document, analyze_repository_documents
from mixed_document import MixedDocumentIngestor
from structured_document import StructuredDocumentIngestor
from unstructured_document import UnstructuredDocumentIngestor





class DocumentIngestionAdapter:
    """
    Adapter class to unify ingestion for mixed, structured, and unstructured documents.
    """

    def __init__(self):
        self.mixed_ingestor = MixedDocumentIngestor()
        self.structured_ingestor = StructuredDocumentIngestor()
        self.unstructured_ingestor = UnstructuredDocumentIngestor()

    def ingest(self, file_path: str) -> dict:
        """
        Classifies the document and ingests it based on its type.
        """
        doc_type, content = classify_document(file_path)
        if doc_type == "mixed":
            ingestion_result = self.mixed_ingestor.ingest_mixed_document(file_path, content)
        elif doc_type == "structured":
            ingestion_result = self.structured_ingestor.ingest_structured_document(file_path, content)
        elif doc_type == "unstructured":
            ingestion_result = self.unstructured_ingestor.ingest_unstructured_document(file_path, "un-structured", content)
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
        return {"classification": doc_type, "content": content, "ingestion_result": ingestion_result}

    def analyze_directory(self, directory_path: str) -> dict:
        """
        Analyzes all documents in the given directory and classifies them.
       
        """
        return analyze_repository_documents(directory_path)


def ingest_directory(adapter: DocumentIngestionAdapter, directory: str):
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
            ingest_result = adapter.ingest(file_path)
            print(f"Ingestion result for {filename}: {ingest_result}")
        except Exception as e:
            print(f"Error ingesting {filename}: {e}")


if __name__ == "__main__":
    adapter = DocumentIngestionAdapter()   
    # Ingest all files in a directory
    ingest_directory(adapter, "docs")