from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentContainer(BaseModel):
    """
    Simple document container for Graphiti ingestion
    
    This class provides a minimal structure for documents that will be processed
    by Graphiti's dynamic schema evolution and entity extraction capabilities.
    No manual entity extraction or schema patterns needed - Graphiti handles this automatically.
    """
    document_id: str = Field(description="Unique document identifier")
    title: str = Field(description="Document title or name")
    content: str = Field(description="Full document content")
    source_path: str = Field(description="Original file path in blob storage")
    created_date: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_graphiti_episode(self) -> Dict[str, Any]:
        """
        Convert document to Graphiti episode format
        
        Returns:
            Dictionary formatted for Graphiti episode ingestion
        """
        return {
            "name": f"document_{self.document_id}",
            "episode_body": self.content,
            "source_description": f"Document from {self.source_path}: {self.title}",
            "reference_time": self.created_date
        }
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
