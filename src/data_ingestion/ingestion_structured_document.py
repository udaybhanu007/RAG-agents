import re
import json
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from .utility_functions import UtilityFunctions
from .ExtractedResponse import ExtractedResponse
import sys
import logging

# Add the src directory to the path to enable absolute imports  
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from ..core.azure_keyvault_manager import get_secret_from_keyvault
    from langchain_openai import AzureChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    from core.azure_keyvault_manager import get_secret_from_keyvault
    from langchain_openai import AzureChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)
@dataclass
class NodeSchema:
    """Schema for graph nodes extracted from structured data"""
    node_type: str
    properties: Dict[str, Any]
    labels: List[str]
    unique_key: str

@dataclass
class RelationshipSchema:
    """Schema for graph relationships extracted from structured data"""
    source_node: str
    target_node: str
    relationship_type: str
    properties: Dict[str, Any]

@dataclass
class CSVSchemaExtraction:
    """Complete schema extraction result for CSV data"""
    nodes: List[NodeSchema]
    relationships: List[RelationshipSchema]
    metadata: Dict[str, Any]

class StructuredDocumentIngestor:
    def __init__(self):
        self._llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize Azure OpenAI LLM for schema extraction"""
        try:
            azure_deployment = get_secret_from_keyvault("AZURE_OPENAI_DEPLOYMENT")
            azure_api_version = get_secret_from_keyvault("AZURE_OPENAI_API_VERSION")
            
            if not azure_deployment or not azure_api_version:
                logger.warning("Azure OpenAI credentials not found. Schema extraction will be limited.")
                return
            
            self._llm = AzureChatOpenAI(
                azure_deployment=azure_deployment,
                api_version=azure_api_version,
                temperature=0.0,
                max_tokens=4000
            )
            logger.info("Azure OpenAI LLM initialized for schema extraction")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._llm = None

    def ingest_structured_document(self, file_path: str, content: str):
        """
        Ingest a structured document (CSV, Excel, etc.) and return ExtractedResponse.
        Uses LLM-based schema extraction for nodes and relationships.
        """
        print(f"[Structured Ingestion] Processing: {file_path}")
        
        # Check if it's a CSV file and process accordingly
        if file_path.lower().endswith('.csv'):
            return self._process_csv_file(file_path, content)
        else:
            # For non-CSV structured files, use existing logic
            return self._process_general_structured_file(file_path, content)
    
    def _process_csv_file(self, file_path: str, content: str) -> ExtractedResponse:
        """Process CSV file using LLM-based schema extraction"""
        try:
            # Read CSV data
            df = pd.read_csv(file_path)
            logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Extract schema using LLM
            schema_extraction = self._extract_csv_schema_with_llm(df, file_path)
            
            # Convert to structured data format
            structured_data = self._convert_schema_to_structured_data(schema_extraction)
            entities = self._convert_nodes_to_entities(schema_extraction.nodes)
            relationships = self._convert_schema_relationships(schema_extraction.relationships)
            
            # Generate metadata
            metadata = self._generate_metadata(file_path, schema_extraction)
            
            # Convert DataFrame to text for full_text
            full_text = self._dataframe_to_text(df)
            
            return ExtractedResponse(
                full_text=full_text,
                unstructured_chunks=[],
                structured_data=structured_data,
                entities=entities,
                relationships=relationships,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            # Fallback to basic processing
            return self._process_general_structured_file(file_path, content)

    def _extract_csv_schema_with_llm(self, df: pd.DataFrame, file_path: str) -> CSVSchemaExtraction:
        """Extract nodes and relationships schema from CSV using LLM"""
        if not self._llm:
            logger.warning("LLM not available, using fallback schema extraction")
            return self._fallback_schema_extraction(df, file_path)
        
        # Prepare CSV sample for LLM analysis
        csv_sample = self._prepare_csv_sample_for_llm(df)
        
        # Create schema extraction prompt
        prompt = self._create_schema_extraction_prompt(csv_sample, file_path)
        
        try:
            # Get LLM response
            messages = [
                SystemMessage(content=self._get_schema_extraction_system_prompt()),
                HumanMessage(content=prompt)
            ]
            
            response = self._llm.invoke(messages)
            
            # Parse LLM response to extract schema
            return self._parse_llm_schema_response(response.content, df)
            
        except Exception as e:
            logger.error(f"LLM schema extraction failed: {e}")
            return self._fallback_schema_extraction(df, file_path)

    def _get_schema_extraction_system_prompt(self) -> str:
        """System prompt for LLM schema extraction"""
        return """You are an expert in graph database design and structured data analysis. Your task is to analyze CSV data and extract a comprehensive schema for nodes and relationships that would be suitable for a knowledge graph.

For each CSV file, identify:
1. **Node Types**: What entities/concepts should become nodes (e.g., Patient, Image, Diagnosis, Study)
2. **Node Properties**: What columns should become properties of each node type
3. **Relationships**: How different nodes relate to each other
4. **Relationship Properties**: What additional data should be stored on relationships

Guidelines:
- Focus on creating meaningful, queryable graph structures
- Identify unique identifiers for nodes
- Consider cardinality (one-to-one, one-to-many, many-to-many relationships)
- Normalize data appropriately (don't duplicate information)
- Use semantic relationship names (e.g., HAS_DIAGNOSIS, BELONGS_TO_PATIENT)

Respond in valid JSON format with the following structure:
{
  "nodes": [
    {
      "node_type": "NodeTypeName",
      "labels": ["Label1", "Label2"],
      "properties": ["column1", "column2"],
      "unique_key": "primary_identifier_column",
      "description": "Brief description of this node type"
    }
  ],
  "relationships": [
    {
      "source_node_type": "SourceNodeType",
      "target_node_type": "TargetNodeType", 
      "relationship_type": "RELATIONSHIP_NAME",
      "properties": ["relationship_property1"],
      "description": "Brief description of this relationship"
    }
  ],
  "metadata": {
    "domain": "medical/research/general",
    "complexity": "simple/medium/complex",
    "recommended_indexes": ["property1", "property2"]
  }
}"""

    def _create_schema_extraction_prompt(self, csv_sample: str, file_path: str) -> str:
        """Create the main prompt for schema extraction"""
        filename = os.path.basename(file_path)
        
        return f"""Please analyze the following CSV data and extract a comprehensive graph database schema.

**File**: {filename}

**CSV Sample Data**:
```
{csv_sample}
```

**Context**: This appears to be a medical/research dataset based on the filename and columns. Please design a graph schema that:

1. Creates meaningful node types for the main entities
2. Establishes clear relationships between entities
3. Preserves important data as node/relationship properties
4. Enables efficient querying for medical research use cases

Focus on:
- Patient/Subject entities and their attributes
- Medical imaging data and findings
- Diagnostic information
- Study/research metadata
- Temporal relationships if applicable

Please provide the schema in the specified JSON format."""

    def _prepare_csv_sample_for_llm(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        """Prepare a representative sample of CSV data for LLM analysis"""
        # Get basic info
        info = f"Columns ({len(df.columns)}): {', '.join(df.columns)}\n"
        info += f"Total rows: {len(df)}\n\n"
        
        # Add column types and sample values
        info += "Column Information:\n"
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            unique_count = df[col].nunique()
            
            # Get sample values (non-null)
            sample_values = df[col].dropna().head(3).tolist()
            sample_str = ", ".join([str(v) for v in sample_values])
            
            info += f"- {col}: {dtype} ({non_null}/{len(df)} non-null, {unique_count} unique) | Sample: {sample_str}\n"
        
        info += "\nFirst few rows:\n"
        info += df.head(max_rows).to_string(index=False)
        
        return info

    def _parse_llm_schema_response(self, response_content: str, df: pd.DataFrame) -> CSVSchemaExtraction:
        """Parse LLM response and convert to CSVSchemaExtraction"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")
            
            schema_data = json.loads(json_match.group())
            
            # Convert to our schema objects
            nodes = []
            for node_data in schema_data.get('nodes', []):
                node = NodeSchema(
                    node_type=node_data['node_type'],
                    properties={prop: self._infer_property_type(df, prop) for prop in node_data.get('properties', [])},
                    labels=node_data.get('labels', [node_data['node_type']]),
                    unique_key=node_data.get('unique_key', '')
                )
                nodes.append(node)
            
            relationships = []
            for rel_data in schema_data.get('relationships', []):
                relationship = RelationshipSchema(
                    source_node=rel_data['source_node_type'],
                    target_node=rel_data['target_node_type'],
                    relationship_type=rel_data['relationship_type'],
                    properties={prop: self._infer_property_type(df, prop) for prop in rel_data.get('properties', [])}
                )
                relationships.append(relationship)
            
            metadata = schema_data.get('metadata', {})
            metadata['llm_generated'] = True
            metadata['schema_version'] = '1.0'
            
            return CSVSchemaExtraction(
                nodes=nodes,
                relationships=relationships,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._fallback_schema_extraction(df, "unknown")

    def _infer_property_type(self, df: pd.DataFrame, column: str) -> str:
        """Infer the property type from DataFrame column"""
        if column not in df.columns:
            return "string"
        
        dtype = df[column].dtype
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        elif pd.api.types.is_float_dtype(dtype):
            return "float"
        elif pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        else:
            return "string"

    def _fallback_schema_extraction(self, df: pd.DataFrame, file_path: str) -> CSVSchemaExtraction:
        """Fallback schema extraction when LLM is not available"""
        filename = os.path.basename(file_path).lower()
        
        # Basic heuristic-based schema for medical data
        if 'patient' in filename or 'medical' in filename or 'chest' in filename:
            return self._create_medical_fallback_schema(df)
        else:
            return self._create_generic_fallback_schema(df)

    def _create_medical_fallback_schema(self, df: pd.DataFrame) -> CSVSchemaExtraction:
        """Create fallback schema for medical data"""
        nodes = []
        relationships = []
        
        # Common medical data patterns
        patient_cols = [col for col in df.columns if any(term in col.lower() for term in ['patient', 'id'])]
        image_cols = [col for col in df.columns if any(term in col.lower() for term in ['image', 'file', 'png', 'jpg'])]
        finding_cols = [col for col in df.columns if any(term in col.lower() for term in ['finding', 'label', 'diagnosis'])]
        
        if patient_cols:
            nodes.append(NodeSchema(
                node_type="Patient",
                properties={col: self._infer_property_type(df, col) for col in patient_cols},
                labels=["Patient", "Person"],
                unique_key=patient_cols[0] if patient_cols else ""
            ))
        
        if image_cols:
            nodes.append(NodeSchema(
                node_type="MedicalImage", 
                properties={col: self._infer_property_type(df, col) for col in image_cols},
                labels=["Image", "MedicalImage"],
                unique_key=image_cols[0] if image_cols else ""
            ))
        
        if finding_cols:
            nodes.append(NodeSchema(
                node_type="Finding",
                properties={col: self._infer_property_type(df, col) for col in finding_cols},
                labels=["Finding", "Diagnosis"],
                unique_key=""
            ))
        
        # Create relationships
        if len(nodes) >= 2:
            relationships.append(RelationshipSchema(
                source_node="Patient",
                target_node="MedicalImage",
                relationship_type="HAS_IMAGE",
                properties={}
            ))
            
            if len(nodes) >= 3:
                relationships.append(RelationshipSchema(
                    source_node="MedicalImage",
                    target_node="Finding", 
                    relationship_type="HAS_FINDING",
                    properties={}
                ))
        
        return CSVSchemaExtraction(
            nodes=nodes,
            relationships=relationships,
            metadata={"domain": "medical", "generated_by": "fallback", "complexity": "simple"}
        )
        
    def _create_generic_fallback_schema(self, df: pd.DataFrame) -> CSVSchemaExtraction:
        """Create generic fallback schema for any CSV"""
        # Create a single node type with all columns
        nodes = [NodeSchema(
            node_type="Record",
            properties={col: self._infer_property_type(df, col) for col in df.columns},
            labels=["Record", "Data"],
            unique_key=df.columns[0] if len(df.columns) > 0 else ""
        )]
        
        return CSVSchemaExtraction(
            nodes=nodes,
            relationships=[],
            metadata={"domain": "general", "generated_by": "fallback", "complexity": "simple"}
        )

    def _convert_schema_to_structured_data(self, schema: CSVSchemaExtraction) -> Dict[str, Any]:
        """Convert schema extraction to structured data format"""
        return {
            "schema": {
                "nodes": [
                    {
                        "type": node.node_type,
                        "labels": node.labels,
                        "properties": node.properties,
                        "unique_key": node.unique_key
                    } for node in schema.nodes
                ],
                "relationships": [
                    {
                        "type": rel.relationship_type,
                        "source": rel.source_node,
                        "target": rel.target_node,
                        "properties": rel.properties
                    } for rel in schema.relationships
                ]
            },
            "metadata": schema.metadata,
            "extraction_method": "llm_based"
        }

    def _convert_nodes_to_entities(self, nodes: List[NodeSchema]) -> List[Dict[str, str]]:
        """Convert node schema to entities format"""
        entities = []
        for node in nodes:
            entities.append({
                "type": "node",
                "category": node.node_type.lower(),
                "value": node.node_type,
                "labels": node.labels,
                "properties": list(node.properties.keys())
            })
        return entities

    def _convert_schema_relationships(self, relationships: List[RelationshipSchema]) -> List[Dict[str, str]]:
        """Convert relationship schema to relationships format"""
        converted = []
        for rel in relationships:
            converted.append({
                "source": rel.source_node,
                "target": rel.target_node,
                "relationship": rel.relationship_type,
                "context": "schema_derived",
                "properties": list(rel.properties.keys())
            })
        return converted

    def _generate_metadata(self, file_path: str, schema: CSVSchemaExtraction) -> Dict[str, Any]:
        """Generate metadata for the extraction"""
        source_filename = os.path.basename(file_path)
        
        # Generate blob URL from environment variables if available
        storage_account = get_secret_from_keyvault("AZURE_STORAGE_ACCOUNT_NAME")
        container_name = get_secret_from_keyvault("AZURE_BLOB_CONTAINER_NAME")
        
        metadata = {
            "file_path": file_path,
            "file_name": source_filename,
            "classification": "structured",
            "document_type": "csv_structured",
            "processing_timestamp": "2025-08-14T00:00:00Z",
            "schema_extraction": {
                "method": "llm_based" if self._llm else "fallback",
                "node_count": len(schema.nodes),
                "relationship_count": len(schema.relationships),
                "domain": schema.metadata.get("domain", "unknown")
            }
        }
        
        # Add blob URL and source info if available
        if storage_account and container_name:
            blob_url = f"https://{storage_account}.blob.core.windows.net/{container_name}/{source_filename}"
            metadata["blob_url"] = blob_url
            metadata["container_name"] = container_name
            metadata["storage_account"] = storage_account
            metadata["source_type"] = "azure_blob"
        else:
            metadata["source_type"] = "local_file"
            
        return metadata

    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to structured text representation"""
        text_parts = []
        
        # Add basic info
        text_parts.append(f"CSV Data Summary:")
        text_parts.append(f"- Rows: {len(df)}")
        text_parts.append(f"- Columns: {len(df.columns)}")
        text_parts.append(f"- Columns: {', '.join(df.columns)}")
        text_parts.append("")
        
        # Add column information
        text_parts.append("Column Information:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            unique_count = df[col].nunique()
            text_parts.append(f"- {col}: {dtype} ({non_null}/{len(df)} non-null, {unique_count} unique values)")
        text_parts.append("")
        
        # Add sample data
        text_parts.append("Sample Data (first 10 rows):")
        text_parts.append(df.head(10).to_string(index=False))
        
        return "\n".join(text_parts)

    def _process_general_structured_file(self, file_path: str, content: str) -> ExtractedResponse:
        """Process non-CSV structured files using original logic"""
        print(f"[General Structured Ingestion] Processing: {file_path}")
        
        full_text = content or ""
        
        # Use simplified extraction for non-CSV files
        structured_data = {
            "file_type": "structured_document",
            "content_summary": f"Structured document with {len(full_text)} characters"
        }
        
        entities = self._extract_basic_entities(full_text)
        relationships = self._extract_basic_relationships(full_text, entities)
        
        metadata = self._generate_basic_metadata(file_path)
        
        return ExtractedResponse(
            full_text=full_text,
            unstructured_chunks=[],
            structured_data=structured_data,
            entities=entities,
            relationships=relationships,
            metadata=metadata
        )

    def _extract_basic_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract basic entities from text"""
        entities = []
        
        # Simple pattern-based extraction
        patterns = {
            "identifier": r'\b[A-Z0-9]{5,}\b',
            "number": r'\b\d+\.?\d*\b',
            "date": r'\b\d{4}-\d{2}-\d{2}\b'
        }
        
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in set(matches):  # Remove duplicates
                entities.append({
                    "type": "extracted",
                    "category": category,
                    "value": match
                })
        
        return entities

    def _extract_basic_relationships(self, text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Extract basic relationships from text"""
        relationships = []
        
        # Simple co-occurrence based relationships
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1["category"] != entity2["category"]:
                    relationships.append({
                        "source": entity1["value"],
                        "target": entity2["value"],
                        "relationship": "co_occurs_with",
                        "context": "basic_extraction"
                    })
        
        return relationships[:50]  # Limit to prevent explosion

    def _generate_basic_metadata(self, file_path: str) -> Dict[str, Any]:
        """Generate basic metadata"""
        source_filename = os.path.basename(file_path)
        
        return {
            "file_path": file_path,
            "file_name": source_filename,
            "classification": "structured",
            "document_type": "general_structured",
            "processing_timestamp": "2025-08-14T00:00:00Z",
            "extraction_method": "basic_pattern",
            "source_type": "local_file"
        }

    def ingest_structured_pdf_document(self, file_path: str, content: str):
        """
        Legacy method for PDF structured documents - redirects to main ingestion
        """
        logger.warning("ingest_structured_pdf_document is deprecated. Use ingest_structured_document instead.")
        return self.ingest_structured_document(file_path, content)

    def get_schema_for_csv(self, file_path: str) -> Optional[CSVSchemaExtraction]:
        """
        Public method to get just the schema extraction for a CSV file
        Useful for preview/validation before full ingestion
        """
        try:
            df = pd.read_csv(file_path)
            return self._extract_csv_schema_with_llm(df, file_path)
        except Exception as e:
            logger.error(f"Failed to extract schema for {file_path}: {e}")
            return None

    def validate_csv_schema(self, file_path: str, expected_node_types: List[str] = None) -> Dict[str, Any]:
        """
        Validate that the extracted schema meets certain criteria
        """
        schema = self.get_schema_for_csv(file_path)
        if not schema:
            return {"valid": False, "error": "Could not extract schema"}
        
        validation_result = {
            "valid": True,
            "node_count": len(schema.nodes),
            "relationship_count": len(schema.relationships),
            "has_unique_keys": all(node.unique_key for node in schema.nodes),
            "domain": schema.metadata.get("domain", "unknown")
        }
        
        if expected_node_types:
            found_types = [node.node_type for node in schema.nodes]
            missing_types = set(expected_node_types) - set(found_types)
            validation_result["missing_node_types"] = list(missing_types)
            validation_result["valid"] = len(missing_types) == 0
        
        return validation_result