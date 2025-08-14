# LLM-Based CSV Schema Extraction for RAG Agents

## Overview

This implementation provides an intelligent, LLM-powered approach to extract graph database schemas from structured CSV data. Instead of manually defining nodes and relationships, the system uses Azure OpenAI to analyze CSV structure and automatically generate optimal graph schemas.

## Key Features

### 🧠 Intelligent Schema Extraction
- **LLM-Powered Analysis**: Uses Azure OpenAI (GPT-4o-mini) to understand CSV structure and domain context
- **Domain-Aware**: Recognizes medical, research, and general data patterns
- **Automatic Node/Relationship Identification**: Extracts meaningful entities and their relationships
- **Property Type Inference**: Automatically determines data types for graph properties

### 📊 Comprehensive Schema Generation
- **Node Types**: Identifies primary entities (Patient, Image, Finding, etc.)
- **Relationships**: Discovers semantic connections between entities
- **Unique Keys**: Identifies primary identifiers for each node type
- **Labels**: Assigns appropriate Neo4j labels for better organization

### 🔄 Fallback Mechanisms
- **Graceful Degradation**: Falls back to heuristic-based extraction if LLM is unavailable
- **Domain-Specific Fallbacks**: Special handling for medical and research data
- **Robust Error Handling**: Continues processing even with partial failures

## Architecture Changes

### Before (Manual Approach)
```python
# Static, hardcoded patterns
medical_patterns = {
    "condition": r'\b(pneumonia|covid-19|tuberculosis)\b',
    "imaging": r'\b(chest\s+x-ray|ct\s+scan)\b'
}

# Manual relationship extraction
relationships = []
for entity in entities:
    if entity["category"] == "algorithm":
        # Hardcoded logic for each entity type
```

### After (LLM-Based Approach)
```python
# Dynamic, context-aware extraction
schema = self._extract_csv_schema_with_llm(df, file_path)

# Intelligent prompt-based analysis
prompt = self._create_schema_extraction_prompt(csv_sample, file_path)
response = self._llm.invoke(messages)
schema = self._parse_llm_schema_response(response.content, df)
```

## Schema Extraction Results

### BBox_List_2017.csv
```json
{
  "nodes": [
    {
      "type": "Image",
      "labels": ["MedicalImage"],
      "properties": ["image_index", "bbox_x", "bbox_y", "bbox_width", "bbox_height"],
      "unique_key": "image_index"
    },
    {
      "type": "Finding", 
      "labels": ["MedicalFinding"],
      "properties": ["finding_label"],
      "unique_key": "finding_label"
    }
  ],
  "relationships": [
    {
      "source": "Image",
      "target": "Finding", 
      "type": "HAS_FINDING",
      "properties": ["bbox_coordinates"]
    }
  ]
}
```

### Data_Entry_2017.csv
```json
{
  "nodes": [
    {
      "type": "Patient",
      "labels": ["Patient"],
      "properties": ["Patient ID", "Patient Age", "Patient Gender"],
      "unique_key": "Patient ID"
    },
    {
      "type": "Image",
      "labels": ["Image"], 
      "properties": ["Image Index", "View Position", "Width", "Height"],
      "unique_key": "Image Index"
    },
    {
      "type": "Finding",
      "labels": ["Finding"],
      "properties": ["Finding Labels"],
      "unique_key": "Finding Labels"
    }
  ],
  "relationships": [
    {
      "source": "Patient",
      "target": "Image",
      "type": "HAS_IMAGE"
    },
    {
      "source": "Image", 
      "target": "Finding",
      "type": "HAS_FINDING"
    }
  ]
}
```

## Usage Examples

### Basic CSV Ingestion
```python
from src.data_ingestion.ingestion_structured_document import StructuredDocumentIngestor

ingester = StructuredDocumentIngestor()
result = ingester.ingest_structured_document("data.csv", "")

print(f"Nodes: {len(result.entities)}")
print(f"Relationships: {len(result.relationships)}")
print(f"Schema: {result.structured_data['schema']}")
```

### Schema Preview
```python
# Get schema without full ingestion
schema = ingester.get_schema_for_csv("data.csv")
if schema:
    print(f"Domain: {schema.metadata['domain']}")
    print(f"Node types: {[node.node_type for node in schema.nodes]}")
    print(f"Relationship types: {[rel.relationship_type for rel in schema.relationships]}")
```

### Schema Validation
```python
validation = ingester.validate_csv_schema("data.csv", 
                                         expected_node_types=["Patient", "Image"])
print(f"Valid: {validation['valid']}")
print(f"Node count: {validation['node_count']}")
print(f"Has unique keys: {validation['has_unique_keys']}")
if 'missing_node_types' in validation:
    print(f"Missing types: {validation['missing_node_types']}")
```

## Configuration

### Azure OpenAI Setup
The system requires these environment variables (configured via Azure Key Vault or .env.dev):
```bash
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
```

### Credential Management
The system uses Azure Key Vault manager for secure credential handling:
- **Production**: Retrieves secrets from Azure Key Vault
- **Development**: Falls back to `.env.dev` when `Keyvalue_Enabled=false`
- **Error Handling**: Graceful degradation if credentials unavailable

### Fallback Behavior
- **LLM Available**: Uses intelligent schema extraction
- **LLM Unavailable**: Falls back to heuristic-based patterns
- **Medical Data**: Special handling for medical domain entities
- **General Data**: Generic schema with basic node/relationship patterns

## Benefits

### 🎯 Precision
- **Context-Aware**: Understands domain-specific terminology and relationships
- **Adaptive**: Adjusts schema based on actual data structure
- **Semantic**: Creates meaningful relationship names (not just "connected_to")

### 🚀 Efficiency
- **Automated**: No manual schema definition required
- **Scalable**: Works with any CSV structure
- **Fast**: Single LLM call per file for complete schema

### 🔧 Maintainability
- **No Hardcoding**: Eliminates static pattern maintenance
- **Extensible**: Easy to add new domains or data types
- **Debuggable**: Clear separation between LLM logic and fallback logic

## Error Handling

### LLM Failure Scenarios
1. **Azure OpenAI Unavailable**: System automatically falls back to heuristic-based schema extraction
2. **Invalid Credentials**: Logs warning and continues with fallback patterns
3. **Rate Limiting**: Built-in retry logic with exponential backoff
4. **Malformed Response**: JSON validation with graceful error recovery

### Data Processing Errors
1. **Invalid CSV**: Pandas error handling with detailed error messages
2. **Missing Columns**: Dynamic property inference based on available data
3. **Empty Files**: Minimal schema generation with appropriate metadata
4. **Encoding Issues**: Automatic encoding detection and conversion

### Schema Validation
1. **Missing Required Fields**: Automatic generation of placeholder values
2. **Invalid Relationships**: Validation and correction of relationship definitions
3. **Type Mismatches**: Intelligent type inference and conversion
4. **Circular Dependencies**: Detection and resolution of circular references

### Logging and Monitoring
- **Structured Logging**: Comprehensive logging for debugging and monitoring
- **Performance Metrics**: Execution time tracking for optimization
- **Error Tracking**: Detailed error messages with context for troubleshooting

## Future Enhancements

- **Multi-CSV Relationships**: Cross-file entity linking
- **Schema Evolution**: Version management for changing schemas
- **Custom Prompts**: Domain-specific prompt templates
- **Validation Rules**: Business logic validation for extracted schemas
- **Performance Caching**: Schema result caching for repeated files

## Testing and Validation

### Schema Testing
```python
# Test schema extraction for a specific file
from src.data_ingestion.ingestion_structured_document import StructuredDocumentIngestor

ingester = StructuredDocumentIngestor()

# Test CSV schema extraction
schema = ingester.get_schema_for_csv("doc-ingestion/BBox_List_2017.csv")
print(f"Extracted {len(schema.nodes)} node types")
print(f"Generated {len(schema.relationships)} relationships")

# Validate schema meets requirements
validation = ingester.validate_csv_schema(
    "doc-ingestion/Data_Entry_2017.csv",
    expected_node_types=["Patient", "Image", "Finding"]
)
print(f"Schema validation: {validation}")
```

### Integration Testing
```python
# Test complete ingestion workflow
result = ingester.ingest_structured_document("test-data.csv", "")
assert len(result.entities) > 0
assert len(result.relationships) > 0
assert result.structured_data['schema'] is not None
```

### Performance Testing
```python
# Test large file processing
import time
start_time = time.time()
result = ingester.ingest_structured_document("large-dataset.csv", "")
processing_time = time.time() - start_time
print(f"Processed large file in {processing_time:.2f} seconds")
```

## Integration Points

The LLM-based schema extraction integrates seamlessly with:

### **Database Integrations**
- **Neo4j Graph Database**: 
  - Uses `Neo4jCSVIngestor` for automated graph ingestion
  - Creates nodes, relationships, and constraints based on extracted schema
  - Supports batch processing for large datasets
- **Qdrant Vector Database**: 
  - Stores semantic embeddings for extracted entities
  - Enables similarity search across medical concepts
  - Integrates with schema metadata for enhanced retrieval

### **Storage Systems**
- **Azure Blob Storage**: 
  - Processes files directly from blob storage
  - Generates blob URLs for metadata tracking
  - Supports distributed file processing workflows
- **Local File System**: 
  - Fallback for development and testing environments
  - Direct file path processing capabilities

### **Security and Configuration**
- **Azure Key Vault**: 
  - Secure credential management for all external services
  - Automatic fallback to environment variables
  - Centralized configuration management
- **Environment Configuration**: 
  - `.env.dev` for development settings
  - Production-ready credential resolution

### **Multi-Agent Workflow**
- **Workflow Orchestration**: 
  - Integrates as part of larger RAG pipeline
  - Supports chaining with other processing agents
  - Provides structured output for downstream consumers
- **State Management**: 
  - Maintains processing state across workflow steps
  - Enables resume/retry capabilities for long-running processes
