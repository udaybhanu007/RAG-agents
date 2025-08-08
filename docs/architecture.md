# RAG-Agents Project Architecture

## Overview

The RAG-Agents project is a sophisticated **Multi-Agent Retrieval-Augmented Generation (RAG) system** designed for medical and research document processing. It combines intelligent document ingestion, multi-modal retrieval (vector + graph), and AI-powered synthesis to provide accurate, contextual answers to complex queries.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG-Agents System                           │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI Rest API Layer                                        │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Agent Workflow Engine (LangGraph)                       │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐  │
│  │Orchestrator │ Vector RAG  │ Graph RAG   │ Validator &     │  │
│  │   Agent     │   Agent     │   Agent     │ Synthesizer     │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Document Ingestion Pipeline                                   │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │ Classification│ Structured   │ Unstructured │ Mixed        │ │
│  │   Engine      │  Ingestion   │  Ingestion   │ Ingestion    │ │
│  └──────────────┴──────────────┴──────────────┴──────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Storage Layer                                                 │
│  ┌──────────────┬──────────────┬──────────────────────────────┐ │
│  │   Qdrant     │    Neo4j     │     Configuration            │ │
│  │Vector Database│Graph Database│   & Environment             │ │
│  └──────────────┴──────────────┴──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
RAG-agents/
├── src/                           # Main source code
│   ├── agents/                    # Multi-agent system
│   │   ├── agents.py             # Core agent implementations
│   │   ├── multi_agent_rag_workflow.py  # LangGraph workflow
│   │   ├── api.py                # FastAPI endpoints
│   │   ├── workflow_state.py     # State management
│   │   ├── observability.py      # Monitoring & metrics
│   │   ├── tool_governance.py    # Security & access control
│   │   ├── validation_synthesis.py # Answer validation & synthesis
│   │   └── logging_config.py     # Structured logging
│   │
│   ├── data_ingestion/            # Document processing pipeline
│   │   ├── document_ingestion_orchestrator.py  # Main orchestrator
│   │   ├── classify_document.py  # Document classification
│   │   ├── mixed_document.py     # Mixed document handling
│   │   ├── ingestion_structured_document.py   # Structured data
│   │   ├── ingestion_unstructured_document.py # Unstructured text
│   │   ├── pdf_extractor.py      # PDF content extraction
│   │   ├── csv_extracter.py      # CSV entity extraction
│   │   ├── retrieval.py          # Search & retrieval
│   │   ├── chunking_unstructured.py # Text chunking
│   │   ├── utility_functions.py  # Helper utilities
│   │   ├── ExtractedResponse.py   # Data models
│   │   └── ingest_to_neo4j.py    # Graph database ingestion
│   │
│   ├── api/                       # API components
│   │   ├── models.py             # API data models
│   │   └── utils.py              # API utilities
│   │
│   ├── core/                      # Core infrastructure
│   │   ├── logging_config.py     # Centralized logging
│   │   └── observability.py      # Performance monitoring
│   │
│   ├── config.py                  # Configuration management
│   └── main.py                    # Application entry point
│
├── tests/                         # Test suites
├── docs/                          # Documentation
├── doc-ingestion/                 # Sample documents for ingestion
├── scripts/                       # Utility scripts
├── docker-compose.yml             # Container orchestration
├── Dockerfile                     # Container definition
└── requirements.txt               # Python dependencies
```

## Core Components

### 1. Multi-Agent Workflow Engine

**Location**: `src/agents/`

#### **OrchestratorAgent**
- **Purpose**: Intelligent query routing and medical relevance validation
- **Key Features**:
  - Medical query validation using LLM
  - Query analysis (intent, entity count, relationships)
  - Routing decisions: `vector`, `graph`, `both`, or `none`
  - Function-calling architecture for tool execution

#### **VectorRAGAgent** 
- **Purpose**: Semantic similarity search with hybrid capabilities
- **Key Features**:
  - Qdrant vector database integration
  - Optional BM25 keyword search for hybrid retrieval
  - Document reranking using LLM
  - Adaptive weighting between vector and keyword search

#### **GraphRAGAgent**
- **Purpose**: Knowledge graph queries for relational information
- **Key Features**:
  - Neo4j graph database integration
  - Entity extraction from queries
  - Dynamic Cypher query generation
  - Multi-scenario query execution

#### **ValidatorAgent & AnswerSynthesisAgent**
- **Purpose**: Result validation and final answer composition
- **Key Features**:
  - Consistency checking between retrieval sources
  - LLM-powered answer synthesis
  - Citation generation and confidence scoring

### 2. Document Ingestion Pipeline

**Location**: `src/data_ingestion/`

#### **Document Classification Engine**
- **File**: `classify_document.py`
- **Purpose**: Intelligent document type detection
- **Classifications**:
  - **Structured**: CSV, Excel, XML, JSON (data-heavy)
  - **Unstructured**: Plain text, narrative documents
  - **Mixed**: PDFs, research papers (both data and narrative)

#### **Ingestion Processors**

##### **StructuredDocumentIngestor**
- **File**: `ingestion_structured_document.py`
- **Handles**: CSV, Excel, structured data
- **Capabilities**:
  - Entity extraction (medical conditions, procedures, imaging)
  - Relationship detection
  - Performance metrics extraction
  - Statistical data parsing

##### **UnstructuredDocumentIngestor**
- **File**: `ingestion_unstructured_document.py`
- **Handles**: Text documents, narrative content
- **Capabilities**:
  - Paragraph-based chunking
  - Vector embedding generation (SentenceTransformers)
  - Qdrant vector store ingestion
  - Batch processing with metadata

##### **MixedDocumentIngestor**
- **File**: `mixed_document.py`
- **Handles**: PDFs, research papers, clinical reports
- **Capabilities**:
  - PDF content extraction (PyMuPDF4LLM)
  - Table extraction and processing
  - Narrative text chunking
  - Entity and relationship extraction

#### **PDF Processing Pipeline**
- **File**: `pdf_extractor.py`
- **Features**:
  - Text extraction with paragraph detection
  - Table extraction and markdown conversion
  - Image extraction and processing
  - Structured markdown output

### 3. Storage Layer

#### **Vector Database (Qdrant)**
- **Purpose**: Semantic similarity search
- **Features**:
  - High-dimensional vector storage
  - Similarity search with configurable thresholds
  - Metadata filtering and payload support
  - Batch ingestion capabilities

#### **Graph Database (Neo4j)**
- **Purpose**: Relationship and entity storage
- **Features**:
  - Medical entity relationships
  - Patient-finding associations
  - Temporal progression tracking
  - Complex Cypher query support

### 4. API Layer

**Location**: `src/agents/api.py`

#### **FastAPI REST API**
- **Endpoints**:
  - `POST /query` - Main query processing
  - `GET /health` - Health checks
  - `GET /metrics` - Prometheus metrics
  - `GET /workflow/info` - Workflow structure info

- **Features**:
  - CORS support for web integration
  - Request/response validation
  - Error handling and logging
  - Prometheus metrics integration

### 5. Infrastructure Components

#### **Observability & Monitoring**
- **File**: `src/agents/observability.py`
- **Features**:
  - Performance measurement per agent
  - Memory usage tracking
  - LangSmith integration for tracing
  - Structured logging with trace IDs

#### **Security & Governance**
- **File**: `src/agents/tool_governance.py`
- **Features**:
  - Role-based access control for tools
  - Agent permission management
  - Secure tool invocation
  - Access denied error handling

#### **Centralized Logging**
- **File**: `src/agents/logging_config.py`
- **Features**:
  - Structured JSON logging
  - Configurable log levels
  - Trace ID correlation
  - Production-ready formatting

## Data Flow

### 1. Document Ingestion Flow

```
Document Input → Classification Engine → Type-Specific Processor
                                      ↓
Mixed Documents → PDF Extractor → Content Separation
                                      ↓
            ┌─────────────────┬─────────────────┐
            ▼                 ▼                 ▼
    Vector Storage    Graph Storage    Metadata Storage
    (Qdrant)         (Neo4j)          (File System)
```

### 2. Query Processing Flow

```
User Query → FastAPI → Orchestrator Agent → Route Decision
                                          ↓
                    ┌─────────────────┬─────────────────┐
                    ▼                 ▼                 ▼
            Vector RAG Agent    Graph RAG Agent    Both Agents
                    ↓                 ↓                 ↓
            Qdrant Search      Neo4j Queries     Combined Results
                    ↓                 ↓                 ↓
                    └─────────────────┼─────────────────┘
                                      ▼
                            Validator Agent → Answer Synthesis → Final Response
```

## Technology Stack

### **Core Frameworks**
- **LangGraph**: Multi-agent workflow orchestration
- **LangChain**: AI/LLM integration and tools
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation and serialization

### **AI/ML Components**
- **Azure OpenAI**: GPT-4 for reasoning and synthesis
- **SentenceTransformers**: Text embeddings
- **HuggingFace**: Alternative embedding models

### **Databases**
- **Qdrant**: Vector similarity search
- **Neo4j**: Graph database for relationships

### **Document Processing**
- **PyMuPDF4LLM**: PDF text extraction
- **PDFPlumber**: Table extraction from PDFs
- **Pandas**: Structured data manipulation

### **Infrastructure**
- **Prometheus**: Metrics and monitoring
- **Structlog**: Structured logging
- **Docker**: Containerization
- **Python-dotenv**: Environment management

## Configuration & Environment

### **Environment Variables**
```bash
# Azure OpenAI
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Qdrant Configuration
QDRANT_API_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=documents

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_JSON_LOGS=true
ENABLE_COLORED_LOGS=false
```

## Deployment Architecture

### **Container Setup**
- **Docker Compose**: Multi-service orchestration
- **Services**: 
  - RAG Application
  - Qdrant Vector DB
  - Neo4j Graph DB
  - Prometheus Monitoring

### **Scalability Considerations**
- **Horizontal Scaling**: Multiple API instances
- **Database Scaling**: Qdrant cluster, Neo4j clustering
- **Caching**: Redis for frequent queries
- **Load Balancing**: Nginx/HAProxy for API distribution

## Security Features

### **Access Control**
- Role-based agent permissions
- Tool-level access restrictions
- Secure tool invocation patterns

### **Data Security**
- Environment-based configuration
- Secure database connections
- API request validation

### **Monitoring & Logging**
- Comprehensive audit trails
- Performance monitoring
- Error tracking and alerting

## Performance Optimizations

### **Ingestion Pipeline**
- Batch processing for vector ingestion
- Parallel document processing
- Efficient chunking strategies

### **Query Processing**
- Adaptive hybrid search weighting
- Smart routing decisions
- Result caching capabilities

### **Database Optimizations**
- Vector index optimization
- Graph query optimization
- Connection pooling

## Future Enhancements

### **Planned Features**
- Advanced entity relationship modeling
- Multi-modal document support (images, audio)
- Real-time collaborative filtering
- Advanced caching strategies

### **Scalability Improvements**
- Microservices architecture
- Event-driven processing
- Distributed caching
- Auto-scaling capabilities

## Getting Started

### **Prerequisites**
- Python 3.9+
- Docker & Docker Compose
- Access to Azure OpenAI
- Qdrant and Neo4j instances

### **Quick Start**
1. Clone the repository
2. Set up environment variables
3. Run `docker-compose up -d`
4. Start document ingestion: `python src/main.py`
5. Query via API: `POST http://localhost:8000/query`

---

*This architecture supports complex medical and research document analysis with intelligent routing, multi-modal retrieval, and comprehensive observability for production deployments.*
