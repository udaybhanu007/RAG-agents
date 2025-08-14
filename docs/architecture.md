# RAG-Agents Project Architecture

## Overview

The RAG-Agents project is a sophisticated **Multi-Agent Retrieval-Augmented Generation (RAG) system** designed for medical and research document processing. It combines intelligent document ingestion, multi-modal retrieval (vector + graph), and AI-powered synthesis to provide accurate, contextual answers to complex queries.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG-Agents System                           │
├─────────────────────────────────────────────────────────────────┤
│  User Interface Layer                                          │
│  ┌──────────────────┬──────────────────────────────────────────┐ │
│  │  Streamlit Web   │       Direct Python API                 │ │
│  │   Interface      │     (main.py entry point)               │ │
│  └──────────────────┴──────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Security & Middleware Layer                                   │
│  ┌─────────────────┬─────────────────┬────────────────────────┐ │
│  │ Azure Key Vault │ Security        │ Input Sanitization     │ │
│  │  Integration    │ Middleware      │ & Validation           │ │
│  └─────────────────┴─────────────────┴────────────────────────┘ │
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
│  Storage & Cloud Services Layer                                │
│  ┌──────────────┬──────────────┬──────────────────────────────┐ │
│  │   Qdrant     │    Neo4j     │    Azure Services            │ │
│  │Vector Database│Graph Database│ (Blob Storage, Key Vault)   │ │
│  └──────────────┴──────────────┴──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
RAG-agents/
├── src/                           # Main source code
│   ├── agents/                    # Multi-agent system
│   │   ├── agents.py             # Core agent implementations (OrchestratorAgent, VectorRAGAgent, GraphRAGAgent)
│   │   ├── multi_agent_rag_workflow.py  # LangGraph workflow orchestration
│   │   ├── streamlit_app.py      # Streamlit web interface
│   │   ├── workflow_state.py     # State management & data flow
│   │   ├── tool_governance.py    # Security & access control
│   │   ├── validation_synthesis.py # Answer validation & synthesis
│   │   └── logging_config.py     # Structured logging configuration
│   │
│   ├── data_ingestion/            # Document processing pipeline
│   │   ├── document_ingestion_orchestrator.py  # Main orchestrator
│   │   ├── classify_document.py  # Document classification engine
│   │   ├── mixed_document.py     # Mixed document handling (PDFs with data+text)
│   │   ├── ingestion_structured_document.py   # Structured data processing
│   │   ├── ingestion_unstructured_document.py # Unstructured text processing
│   │   ├── pdf_extractor.py      # PDF content extraction
│   │   ├── csv_extracter.py      # CSV entity extraction
│   │   ├── chunking_unstructured.py # Text chunking strategies
│   │   ├── utility_functions.py  # Helper utilities
│   │   ├── ExtractedResponse.py   # Data models & schemas
│   │   └── ingest_to_neo4j.py    # Graph database ingestion
│   │
│   ├── core/                      # Core infrastructure & security
│   │   ├── azure_keyvault_manager.py  # Azure Key Vault integration
│   │   ├── security_middleware.py     # Security middleware layer
│   │   ├── security_validator.py      # Input validation & sanitization
│   │   ├── input_sanitization.py      # Advanced input sanitization
│   │   ├── logging_config.py          # Centralized logging
│   │   └── observability.py           # Performance monitoring & tracing
│   │
│   └── main.py                    # Application entry point & document ingestion
│
├── docs/                          # Documentation
│   ├── architecture.md           # System architecture (this document)
│   └── security_practices.md     # Security implementation guide
├── doc-ingestion/                 # Sample documents for ingestion
│   ├── ARXIV_V5_CHESTXRAY.pdf    # Medical research papers
│   ├── BBox_List_2017.csv        # Structured medical data
│   ├── Data_Entry_2017.csv       # Patient data samples
│   └── *.pdf                     # Additional medical documents
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
  - Routing decisions: `vector`, `graph`, `both` (sequential execution), or `none`
  - Function-calling architecture for tool execution
  - Smart routing logic to determine optimal retrieval strategy

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
  - LLM-powered answer synthesis from single or multiple agent results
  - Citation generation and confidence scoring
  - Handling of sequential agent execution results (when routing is "both")
  - Quality assurance for combined vector and graph retrieval outputs

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

### 4. User Interface Layer

**Location**: `src/agents/streamlit_app.py`

#### **Streamlit Web Interface**
- **Purpose**: Interactive web-based frontend for the RAG system
- **Key Features**:
  - Professional responsive design with custom CSS styling
  - Real-time query processing with status indicators
  - Example queries for medical and research scenarios
  - Session state management and error handling
  - Live workflow initialization and health monitoring
  - Query history and result caching

#### **Direct Python Integration**
- **File**: `src/main.py`
- **Purpose**: Direct programmatic access and batch processing
- **Features**:
  - Document ingestion from Azure Blob Storage or local directories
  - Command-line interface for bulk operations
  - Integration testing and workflow validation

### 5. Security & Infrastructure Layer

**Location**: `src/core/`

#### **Azure Key Vault Integration**
- **File**: `azure_keyvault_manager.py`
- **Purpose**: Centralized secrets management using Azure Key Vault
- **Features**:
  - Azure CLI authentication with DefaultAzureCredential
  - Fallback to `.env.dev` files for development environments
  - Configurable Key Vault enablement via `Keyvalue_Enabled` flag
  - Secure retrieval of API keys, connection strings, and sensitive data
  - Environment-specific configuration management

#### **Security Middleware & Validation**
- **Files**: `security_middleware.py`, `security_validator.py`, `input_sanitization.py`
- **Purpose**: Multi-layered security protection
- **Features**:
  - Advanced input sanitization and validation
  - Prompt injection attack prevention
  - Query security validation with configurable policies
  - Secure agent communication patterns
  - Role-based access control enforcement

#### **Observability & Monitoring**
- **File**: `observability.py`
- **Purpose**: Comprehensive system monitoring and tracing
- **Features**:
  - LangSmith integration for AI workflow tracing
  - Performance measurement per agent with timing metrics
  - Memory usage tracking and resource monitoring
  - Structured logging with correlation IDs
  - Custom metrics for workflow performance analysis

## Data Flow

### 1. Document Ingestion Flow

```
Document Input → Azure Blob Storage / Local Directory → Classification Engine
                                                               ↓
                               Type-Specific Processor Selection
                                                               ↓
        ┌──────────────────────┬──────────────────────┬──────────────────────┐
        ▼                      ▼                      ▼                      ▼
  Structured Data         Unstructured Text      Mixed Documents       PDF Extraction
   (CSV, Excel)           (Plain Text)          (Research Papers)     (Tables + Text)
        ↓                      ↓                      ↓                      ↓
Entity Extraction      Text Chunking         Content Separation      Multimodal Processing
        ↓                      ↓                      ↓                      ↓
    ┌───────────────────────────┼──────────────────────────┼───────────────────────┐
    ▼                          ▼                          ▼                       ▼
Vector Storage             Graph Storage           Metadata Storage      Azure Blob Storage
 (Qdrant)                  (Neo4j)               (File System)         (Document Archive)
```

### 2. Query Processing Flow

```
User Input → Streamlit Interface / Direct API → Security Middleware
                                                       ↓
                            Query Sanitization & Validation
                                                       ↓
                              Orchestrator Agent → Route Decision
                                                       ↓
                    ┌─────────────────┬─────────────────┬─────────────────┐
                    ▼                 ▼                 ▼                 ▼
            "vector" only      "graph" only       "both"              "none"
                    ↓                 ↓                 ↓                 ↓
            Vector RAG Agent    Graph RAG Agent   Vector → Graph     No Retrieval
                    ↓                 ↓            (Sequential)           ↓
            Qdrant Search      Neo4j Queries    Combined Results    Direct Response
                    ↓                 ↓                 ↓                 ↓
                    └─────────────────┼─────────────────┼─────────────────┘
                                      ▼
            Validator Agent → Answer Synthesis → Response + Citations
                                      ↓
                              LangSmith Tracing & Logging
```

## Technology Stack

### **Core Frameworks**
- **LangGraph**: Multi-agent workflow orchestration and state management
- **LangChain**: AI/LLM integration, tools, and document processing
- **Streamlit**: Interactive web interface and user experience
- **Pydantic**: Data validation, serialization, and type safety

### **AI/ML Components**
- **Azure OpenAI**: GPT-4o-mini for reasoning, analysis, and synthesis
- **SentenceTransformers**: High-quality text embeddings for semantic search
- **LangSmith**: AI workflow tracing, debugging, and performance monitoring

### **Databases & Storage**
- **Qdrant**: High-performance vector similarity search and semantic retrieval
- **Neo4j**: Graph database for complex relationships and entity connections
- **Azure Blob Storage**: Document storage and archival with secure access

### **Document Processing**
- **PyMuPDF4LLM**: Advanced PDF text extraction with layout preservation
- **PDFPlumber**: Precise table extraction and structured data parsing
- **Pandas**: Structured data manipulation and analysis
- **Pillow**: Image processing and extraction from documents

### **Cloud & Security**
- **Azure Key Vault**: Centralized secrets management and secure configuration
- **Azure Identity**: Authentication and access control integration
- **Azure Storage**: Scalable blob storage with secure access patterns

### **Infrastructure & Monitoring**
- **Structlog**: Structured logging with correlation and tracing
- **LangSmith**: Comprehensive observability for AI workflows
- **Python-dotenv**: Environment configuration management

## Configuration & Environment

### **Environment Variables**
```bash
# Azure Key Vault Configuration
AZURE_KEY_VAULT_URL=https://your-keyvault.vault.azure.net/
Keyvalue_Enabled=true  # Set to 'false' for local development

# Azure OpenAI (stored in Key Vault or .env.dev)
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

# Qdrant Configuration (stored in Key Vault or .env.dev)
QDRANT_API_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION=documents

# Neo4j Configuration (stored in Key Vault or .env.dev)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Azure Storage (stored in Key Vault or .env.dev)
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
AZURE_STORAGE_ACCOUNT_KEY=your_storage_key
AZURE_BLOB_CONTAINER_NAME=rag-agents-container

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_JSON_LOGS=true
ENABLE_COLORED_LOGS=false

# LangSmith Tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

### **Security Configuration**
The system uses a dual-mode configuration approach:
- **Production Mode**: `Keyvalue_Enabled=true` - All secrets retrieved from Azure Key Vault
- **Development Mode**: `Keyvalue_Enabled=false` - Secrets loaded from `.env.dev` file

This ensures secure secret management in production while maintaining development flexibility.

## Deployment Architecture

### **Current Implementation**
- **Streamlit Web Application**: Interactive frontend with real-time query processing
- **Azure Cloud Integration**: Key Vault for secrets, Blob Storage for documents
- **Local Development**: Environment-based configuration with fallback mechanisms
- **Standalone Deployment**: Single Python application with integrated components

### **Database Requirements**
- **Qdrant Vector DB**: For semantic search and document retrieval
- **Neo4j Graph DB**: For entity relationships and knowledge graphs
- **Azure Blob Storage**: For document storage and archival

### **Scalability Considerations**
- **Horizontal Scaling**: Multiple Streamlit instances with load balancing
- **Database Scaling**: Qdrant cluster deployment, Neo4j clustering
- **Caching Strategy**: Query result caching with session state management
- **Azure Auto-scaling**: Serverless functions for document processing

## Security Features

### **Multi-Layered Security Architecture**
- **Azure Key Vault Integration**: Centralized secrets management with Azure CLI authentication
- **Input Sanitization**: Advanced query validation and prompt injection prevention
- **Security Middleware**: Request validation and security violation detection
- **Role-Based Access**: Agent permission management and tool governance

### **Data Protection**
- **Azure Managed Identity**: Secure service-to-service authentication
- **Environment Isolation**: Separate configurations for development and production
- **Secure Communication**: Encrypted connections to all external services
- **Audit Logging**: Comprehensive security event tracking

### **Compliance & Monitoring**
- **LangSmith Integration**: AI workflow tracing with data privacy controls
- **Structured Logging**: Correlation IDs and security event tracking
- **Performance Monitoring**: Real-time system health and usage metrics
- **Error Handling**: Graceful degradation and security violation responses

## Performance Optimizations

### **Ingestion Pipeline Optimizations**
- **Azure Blob Storage Integration**: Scalable document storage with automatic ingestion
- **Parallel Document Processing**: Multi-threaded processing for large document sets
- **Intelligent Document Classification**: Automatic routing to appropriate processors
- **Batch Vector Ingestion**: Efficient embedding generation and storage
- **Streaming Document Processing**: Real-time ingestion for continuous updates

### **Query Processing Optimizations**
- **Adaptive Hybrid Search**: Dynamic weighting between vector and keyword search
- **Smart Agent Routing**: Intelligent decisions to minimize processing overhead
- **Sequential Agent Execution**: Efficient orchestration when both vector and graph retrieval are needed
- **Result Caching**: Session-based caching for repeated queries
- **LangSmith Tracing**: Performance monitoring and optimization insights

### **Database & Storage Optimizations**
- **Qdrant Collection Management**: Optimized vector indexing and search parameters
- **Neo4j Query Optimization**: Efficient Cypher queries for graph traversal
- **Azure Blob Tiering**: Cost-effective storage with intelligent data lifecycle
- **Connection Pooling**: Efficient database connection management

## Future Enhancements

### **Planned Features**
- **Enhanced Security Implementation**: Complete Azure native security roadmap
- **Advanced Multi-Modal Support**: Image analysis integration with text processing
- **Real-Time Collaboration**: Multi-user support with shared workspaces
- **Advanced Analytics Dashboard**: Query analytics and usage insights
- **API Gateway Implementation**: RESTful API layer for external integrations
- **Containerization**: Docker support for scalable deployments

### **Azure Native Enhancements**
- **Azure Functions Integration**: Serverless document processing pipelines
- **Azure Cognitive Services**: Enhanced OCR and document understanding
- **Azure Application Insights**: Advanced application performance monitoring
- **Azure API Management**: Professional API gateway with rate limiting
- **Azure Container Instances**: Scalable container deployments

### **AI & ML Improvements**
- **Fine-Tuned Embeddings**: Domain-specific medical embeddings
- **Advanced RAG Techniques**: Retrieval augmentation with reasoning
- **Multi-Agent Collaboration**: Enhanced agent communication protocols
- **Adaptive Learning**: Query performance optimization through feedback

## Getting Started

### **Prerequisites**
- Python 3.9+ with pip package manager
- Azure CLI configured for authentication
- Access to Azure OpenAI service
- Qdrant vector database (local or cloud)
- Neo4j graph database (local or cloud)
- Azure Key Vault (for production) or `.env.dev` file (for development)

### **Quick Start**
1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd RAG-agents
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   - For Development: Create `.env.dev` with all required secrets
   - For Production: Configure Azure Key Vault and set `Keyvalue_Enabled=true`

3. **Initialize Databases**
   - Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
   - Start Neo4j: `docker run -p 7474:7474 -p 7687:7687 neo4j`

4. **Ingest Documents**
   ```bash
   cd src
   python main.py  # Processes documents from Azure Blob Storage or local directory
   ```

5. **Launch Web Interface**
   ```bash
   cd src/agents
   streamlit run streamlit_app.py
   ```

6. **Access Application**
   - Web Interface: `http://localhost:8501`
   - Query medical documents and research papers through the interactive interface

### **Development Setup**
- Use `.env.dev` for local development with `Keyvalue_Enabled=false`
- Configure local databases for testing
- Enable LangSmith tracing for debugging workflows
- Use Streamlit's development mode for real-time code updates

---

*This architecture supports sophisticated medical and research document analysis through intelligent multi-agent orchestration, robust security implementations, and comprehensive Azure cloud integration. The system provides enterprise-grade observability, scalable processing capabilities, and an intuitive web interface for production deployments in healthcare and research environments.*

**Key Differentiators:**
- **Security-First Design**: Azure Key Vault integration with multi-layered validation
- **Production-Ready**: Comprehensive logging, monitoring, and error handling
- **User-Friendly**: Streamlit interface with professional styling and real-time feedback
- **Cloud-Native**: Deep Azure integration for scalable, secure deployments
- **Medical Domain Focus**: Specialized for healthcare documents and research papers
