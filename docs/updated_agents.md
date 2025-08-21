# Enhanced Agentic RAG System - Updated Agents Documentation

## Overview

The Enhanced Agentic RAG System is a sophisticated multi-agent architecture that combines LangGraph workflow orchestration with intelligent reasoning capabilities. This system provides autonomous decision-making for medical document retrieval and analysis using Azure OpenAI, Qdrant vector database, and Neo4j graph database.

## 🏗️ Architecture Overview

### Core Components

```
📁 src/updated_agents/
├── 🤖 simple_agentic_app.py           # Main application orchestrator
├── 🔍 enhanced_query_analyzer.py      # Advanced query analysis
├── 🛠️ dynamic_tool_selector.py        # Intelligent tool selection
├── 📋 execution_planner.py            # Workflow planning engine
├── 🔄 langgraph_agentic_workflow.py   # LangGraph workflow engine
├── 👥 simple_agentic_agents.py        # Multi-agent system
├── 🏗️ base_classes.py                 # Core data structures
├── 🌐 simple_agentic_streamlit.py     # Web interface
└── 💻 working_main.py                 # CLI interface
```

## 📊 System Architecture

### 🏗️ **Complete System Architecture Diagram**

```mermaid
graph TB
    %% User Interface Layer
    subgraph "🌐 User Interface Layer"
        UI[🌐 Streamlit Web Interface<br/>simple_agentic_streamlit.py<br/>- Clean production UI<br/>- Session state management<br/>- Real-time query processing]
        CLI[💻 CLI Interface<br/>working_main.py<br/>- Direct API access<br/>- Testing interface<br/>- Configuration validation]
    end

    %% Security & Configuration Layer
    subgraph "🔐 Security & Configuration Layer"
        KV[🔐 Azure Key Vault<br/>azure_keyvault_manager.py<br/>- Secure secret storage<br/>- Azure CLI authentication<br/>- Environment fallback]
        ENV[📄 Environment Config<br/>.env / .env.dev<br/>- Development settings<br/>- Production overrides<br/>- Key Vault toggles]
        SEC[🛡️ Input Sanitization<br/>input_sanitization.py<br/>- Prompt injection detection<br/>- Input validation<br/>- Output sanitization]
        OBS[📊 Observability<br/>observability.py<br/>- LangSmith integration<br/>- Performance monitoring<br/>- Structured logging]
    end

    %% Core Application Layer
    subgraph "🤖 Core Application Layer"
        APP[🤖 Enhanced Agentic RAG Application<br/>simple_agentic_app.py<br/>- System orchestration<br/>- Component integration<br/>- Health monitoring]
        QA[🔍 Enhanced Query Analyzer<br/>enhanced_query_analyzer.py<br/>- Medical domain validation<br/>- Intent classification<br/>- Complexity assessment]
        TS[🛠️ Dynamic Tool Selector<br/>dynamic_tool_selector.py<br/>- Context-aware selection<br/>- Performance optimization<br/>- Adaptive routing]
        EP[📋 Execution Planner<br/>execution_planner.py<br/>- Step sequencing<br/>- Resource allocation<br/>- Contingency planning]
    end

    %% LangGraph Workflow Engine
    subgraph "🔄 LangGraph Workflow Engine"
        LG[🔄 LangGraph Agentic Workflow<br/>langgraph_agentic_workflow.py<br/>- State-based execution<br/>- Node orchestration<br/>- Conditional routing]
        STATE[📊 Workflow State Management<br/>base_classes.py<br/>- WorkflowState schema<br/>- Agent interfaces<br/>- Data structures]
    end

    %% Multi-Agent System
    subgraph "👥 Multi-Agent System (simple_agentic_agents.py)"
        ORCH[🎭 Agentic Orchestrator Agent<br/>- Medical validation (LLM)<br/>- Query analysis & routing<br/>- Learning memory system<br/>- Performance tracking]
        VRAG[📚 Agentic Vector RAG Agent<br/>- Semantic search (Qdrant)<br/>- Hybrid retrieval strategies<br/>- Adaptive parameter tuning<br/>- Document reranking]
        GRAG[🕸️ Agentic Graph RAG Agent<br/>- Entity extraction<br/>- Dynamic Cypher queries<br/>- Relationship analysis<br/>- Constraint application]
        VAL[✅ Validator Agent<br/>- Result validation<br/>- Consistency checking<br/>- Quality assessment<br/>- Confidence scoring]
        SYN[📝 Answer Synthesis Agent<br/>- Multi-source synthesis<br/>- Citation generation<br/>- Medical context preservation<br/>- Final answer formatting]
    end

    %% External Services & Storage Layer
    subgraph "☁️ External Services & Storage"
        AZURE[☁️ Azure OpenAI<br/>GPT-4o-mini<br/>- LLM reasoning<br/>- Medical validation<br/>- Answer synthesis]
        QDRANT[🔍 Qdrant Vector DB<br/>- Semantic embeddings<br/>- Vector similarity search<br/>- BM25 hybrid search<br/>- Collection: medical_research_doc]
        NEO4J[🕸️ Neo4j Graph DB<br/>- Knowledge graph<br/>- Patient relationships<br/>- Medical entity connections<br/>- Cypher query execution]
        LANGSMITH[📊 LangSmith<br/>- Tracing & debugging<br/>- Performance analytics<br/>- Workflow monitoring<br/>- Error tracking]
    end

    %% Data Processing Pipeline
    subgraph "📥 Data Processing Pipeline"
        INGEST[📥 Document Ingestion<br/>document_ingestion_orchestrator.py<br/>- Medical document processing<br/>- CSV data extraction<br/>- Multi-format support]
        EMBED[🧮 Embeddings Generation<br/>- Vector creation<br/>- Semantic encoding<br/>- Batch processing]
        GRAPH[🔗 Graph Construction<br/>- Entity extraction<br/>- Relationship mapping<br/>- Knowledge graph building]
    end

    %% Connection Flows
    UI --> APP
    CLI --> APP
    
    KV --> APP
    ENV --> KV
    SEC --> APP
    OBS --> APP
    OBS --> LANGSMITH
    
    APP --> QA
    APP --> TS
    APP --> EP
    APP --> LG
    
    LG --> STATE
    LG --> ORCH
    
    ORCH --> VRAG
    ORCH --> GRAG
    VRAG --> VAL
    GRAG --> VAL
    VAL --> SYN
    
    VRAG --> QDRANT
    GRAG --> NEO4J
    ORCH --> AZURE
    VAL --> AZURE
    SYN --> AZURE
    
    INGEST --> EMBED
    INGEST --> GRAPH
    EMBED --> QDRANT
    GRAPH --> NEO4J

    %% Styling
    classDef userLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef securityLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef coreLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef workflowLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef agentLayer fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef storageLayer fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef dataLayer fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class UI,CLI userLayer
    class KV,ENV,SEC,OBS securityLayer
    class APP,QA,TS,EP coreLayer
    class LG,STATE workflowLayer
    class ORCH,VRAG,GRAG,VAL,SYN agentLayer
    class AZURE,QDRANT,NEO4J,LANGSMITH storageLayer
    class INGEST,EMBED,GRAPH dataLayer
```

### 🔄 **Agent Interaction Flow Diagram**

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant App as Enhanced Agentic App
    participant LG as LangGraph Workflow
    participant Analyzer as Query Analyzer
    participant Selector as Tool Selector
    participant Planner as Execution Planner
    participant Orch as Orchestrator Agent
    participant Vec as Vector RAG Agent
    participant Graph as Graph RAG Agent
    participant Val as Validator Agent
    participant Syn as Synthesis Agent
    participant Azure as Azure OpenAI
    participant Qdrant as Qdrant Vector DB
    participant Neo4j as Neo4j Graph DB

    User->>UI: Submit Medical Query
    UI->>App: process_query(query)
    App->>LG: Initialize Workflow
    
    Note over LG: Step 1: Query Analysis
    LG->>Analyzer: comprehensive_query_analysis()
    Analyzer->>Azure: Medical Validation LLM Call
    Azure-->>Analyzer: Validation Result
    Analyzer->>Azure: Intent Classification LLM Call
    Azure-->>Analyzer: Intent & Complexity
    Analyzer-->>LG: Analysis Complete
    
    Note over LG: Step 2: Tool Selection
    LG->>Selector: select_tools(analysis)
    Selector-->>LG: Selected Tools [vector/graph/both]
    
    Note over LG: Step 3: Execution Planning
    LG->>Planner: create_execution_plan()
    Planner-->>LG: Comprehensive Plan
    
    Note over LG: Step 4: Agent Orchestration
    LG->>Orch: execute_plan()
    
    alt Vector Search Route
        Orch->>Vec: execute_vector_search()
        Vec->>Azure: Strategy Reasoning LLM
        Azure-->>Vec: Search Strategy
        Vec->>Qdrant: Semantic/BM25/Hybrid Search
        Qdrant-->>Vec: Relevant Documents
        Vec-->>Orch: Vector Results
    end
    
    alt Graph Search Route
        Orch->>Graph: execute_graph_search()
        Graph->>Azure: Entity Extraction LLM
        Azure-->>Graph: Extracted Entities
        Graph->>Neo4j: Dynamic Cypher Query
        Neo4j-->>Graph: Graph Results
        Graph-->>Orch: Graph Results
    end
    
    Note over Orch: Results Collected
    Orch->>Val: validate_results()
    Val->>Azure: Consistency Validation LLM
    Azure-->>Val: Validation Score
    Val-->>Orch: Validation Complete
    
    Note over Orch: Final Synthesis
    Orch->>Syn: synthesize_answer()
    Syn->>Azure: Answer Generation LLM
    Azure-->>Syn: Final Answer + Citations
    Syn-->>Orch: Synthesis Complete
    
    Orch-->>LG: Execution Complete
    LG-->>App: Workflow Results
    App-->>UI: Formatted Response
    UI-->>User: Display Answer + Sources
```

### 🧠 **Agent Decision Tree Diagram**

```mermaid
flowchart TD
    START([User Query]) --> MED_VAL{Medical Validation}
    
    MED_VAL -->|Valid| INTENT{Query Intent Analysis}
    MED_VAL -->|Invalid| REJECT[Reject: Non-Medical Query]
    
    INTENT -->|Document| DOC_ROUTE[Vector Search Route]
    INTENT -->|Relational| REL_ROUTE[Graph Search Route]
    INTENT -->|Hybrid| BOTH_ROUTE[Sequential Both Routes]
    INTENT -->|General| DIRECT_ROUTE[Direct LLM Response]
    
    DOC_ROUTE --> VEC_STRATEGY{Vector Strategy}
    VEC_STRATEGY -->|Conceptual| SEMANTIC[Semantic Search Only]
    VEC_STRATEGY -->|Specific Term| BM25[BM25 Keyword Search]
    VEC_STRATEGY -->|Complex| HYBRID[Hybrid Vector + BM25]
    
    REL_ROUTE --> ENTITY_EXT[Entity Extraction]
    ENTITY_EXT --> CYPHER_GEN[Dynamic Cypher Generation]
    CYPHER_GEN --> NEO4J_QUERY[Neo4j Query Execution]
    
    BOTH_ROUTE --> VEC_FIRST[Vector Search First]
    VEC_FIRST --> GRAPH_SECOND[Graph Search Second]
    GRAPH_SECOND --> MERGE_RESULTS[Merge & Validate Results]
    
    SEMANTIC --> VALIDATE
    BM25 --> VALIDATE
    HYBRID --> VALIDATE
    NEO4J_QUERY --> VALIDATE
    MERGE_RESULTS --> VALIDATE
    DIRECT_ROUTE --> VALIDATE
    
    VALIDATE{Validation Check} -->|Pass| SYNTHESIZE[Answer Synthesis]
    VALIDATE -->|Fail| FALLBACK[Fallback Strategy]
    
    SYNTHESIZE --> FINAL_ANSWER[Final Answer + Citations]
    FALLBACK --> RETRY[Retry with Different Strategy]
    RETRY --> INTENT
    
    FINAL_ANSWER --> END([Response to User])
    REJECT --> END
    
    %% Styling
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef process fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef route fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class START,END startEnd
    class MED_VAL,INTENT,VEC_STRATEGY,VALIDATE decision
    class DOC_ROUTE,REL_ROUTE,BOTH_ROUTE,DIRECT_ROUTE route
    class ENTITY_EXT,CYPHER_GEN,VEC_FIRST,GRAPH_SECOND,SYNTHESIZE process
    class REJECT,FALLBACK error
```

### 🏗️ **Component Interaction Matrix**

```mermaid
graph TD
    subgraph "Input Processing Layer"
        Q[Query Input] --> SA[Input Sanitization]
        SA --> MV[Medical Validation]
        MV --> QA[Query Analysis]
    end
    
    subgraph "Decision Layer"
        QA --> TS[Tool Selection]
        TS --> EP[Execution Planning]
        EP --> OR[Orchestration Routing]
    end
    
    subgraph "Execution Layer"
        OR --> VR[Vector Retrieval]
        OR --> GR[Graph Retrieval]
        OR --> DR[Direct Response]
        
        VR --> QD[(Qdrant DB)]
        GR --> ND[(Neo4j DB)]
        
        VR --> RV[Results Validation]
        GR --> RV
        DR --> RV
    end
    
    subgraph "Output Processing Layer"
        RV --> AS[Answer Synthesis]
        AS --> CF[Citation Formatting]
        CF --> FR[Final Response]
    end
    
    subgraph "Supporting Services"
        AZ[Azure OpenAI] -.-> MV
        AZ -.-> QA
        AZ -.-> VR
        AZ -.-> GR
        AZ -.-> DR
        AZ -.-> RV
        AZ -.-> AS
        
        LS[LangSmith] -.-> QA
        LS -.-> TS
        LS -.-> EP
        LS -.-> OR
        LS -.-> RV
        LS -.-> AS
        
        KV[Key Vault] -.-> AZ
        KV -.-> QD
        KV -.-> ND
        KV -.-> LS
    end
    
    %% Styling
    classDef inputLayer fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef decisionLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef executionLayer fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef outputLayer fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef servicesLayer fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef database fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    
    class Q,SA,MV,QA inputLayer
    class TS,EP,OR decisionLayer
    class VR,GR,DR,RV executionLayer
    class AS,CF,FR outputLayer
    class AZ,LS,KV servicesLayer
    class QD,ND database
```

### 1. **Enhanced Agentic RAG Application** (`simple_agentic_app.py`)

**Purpose**: Main application orchestrator that coordinates all components

**Key Features**:
- System initialization and health monitoring
- Component integration and dependency management
- Configuration management (Azure Key Vault integration)
- Error handling and recovery mechanisms

**Main Methods**:
```python
class EnhancedAgenticRAGApplication:
    def initialize_system() -> Dict[str, Any]
    def process_query(query: str) -> Dict[str, Any]
    def get_system_status() -> Dict[str, Any]
```

**Initialization Flow**:
1. Load configuration from Azure Key Vault or environment
2. Initialize Azure OpenAI LLM components
3. Set up vector and graph database connections
4. Initialize all agent components
5. Validate system readiness

### 2. **Enhanced Query Analyzer** (`enhanced_query_analyzer.py`)

**Purpose**: Advanced query understanding and classification

**Key Capabilities**:
- **Medical Domain Validation**: Ensures queries are medically relevant
- **Intent Detection**: Classifies queries as document, relational, or hybrid
- **Complexity Assessment**: Evaluates query complexity for routing decisions
- **Entity Recognition**: Identifies medical entities and relationships

**Analysis Pipeline**:
```python
def comprehensive_query_analysis(query: str) -> AnalysisResult:
    # 1. Medical relevance validation
    # 2. Intent classification (document/relational/hybrid)
    # 3. Complexity scoring
    # 4. Entity extraction
    # 5. Confidence assessment
```

**Query Classification Examples**:
- **Document Intent**: "What is NIH Chest X-ray?" → Vector search
- **Relational Intent**: "Total male patients age 17 with effusion?" → Graph search
- **Hybrid Intent**: "Pneumonia cases and treatment protocols?" → Both search types

### 3. **Dynamic Tool Selector** (`dynamic_tool_selector.py`)

**Purpose**: Intelligent selection of retrieval tools based on query analysis

**Selection Strategy**:
- **Vector Search**: For conceptual queries and document retrieval
- **Graph Search**: For relational queries with specific entity relationships  
- **Hybrid Search**: For complex queries requiring both approaches
- **No Retrieval**: For general medical knowledge queries

**Tool Selection Logic**:
```python
def select_tools(analysis: AnalysisResult) -> List[str]:
    if analysis.intent == "relational" and analysis.has_relationships:
        return ["graph_search"]
    elif analysis.intent == "document":
        return ["vector_search"]
    elif analysis.complexity == "high":
        return ["vector_search", "graph_search"]
    else:
        return ["vector_search"]
```

### 4. **Execution Planner** (`execution_planner.py`)

**Purpose**: Creates comprehensive execution plans with contingencies

**Planning Features**:
- **Step Sequencing**: Optimizes execution order for efficiency
- **Resource Allocation**: Manages computational resources
- **Contingency Planning**: Provides fallback strategies
- **Performance Estimation**: Predicts execution time and resource usage

**Plan Structure**:
```python
class ExecutionPlan:
    plan_id: str
    steps: List[ExecutionStep]
    contingencies: List[ContingencyPlan]
    estimated_duration: str
    resource_requirements: Dict[str, Any]
```

### 5. **LangGraph Agentic Workflow** (`langgraph_agentic_workflow.py`)

**Purpose**: State-based workflow orchestration using LangGraph

**Workflow Nodes**:
- **Query Analysis**: Enhanced query understanding
- **Tool Selection**: Dynamic tool selection
- **Execution Planning**: Comprehensive plan creation
- **Agent Orchestration**: Multi-agent execution
- **Validation**: Result validation and quality checks
- **Synthesis**: Final answer generation

**State Management**:
```python
class WorkflowState:
    query: str
    analysis_result: AnalysisResult
    selected_tools: List[str]
    execution_plan: ExecutionPlan
    agent_results: Dict[str, Any]
    validation_result: ValidationResult
    final_answer: str
```

## 👥 Multi-Agent System (`simple_agentic_agents.py`)

### Agent Architecture

The system employs five specialized agents, each with distinct responsibilities:

### 1. **Agentic Orchestrator Agent**

**Role**: Central coordinator and decision maker

**Capabilities**:
- **Medical Validation**: LLM-powered validation of medical query relevance
- **Query Analysis**: Deep understanding of query intent and complexity
- **Route Decisions**: Intelligent routing to appropriate specialist agents
- **Learning Memory**: Continuous learning from past interactions

**Key Methods**:
```python
def enhanced_reasoning(query: str) -> Dict[str, Any]
def medical_validation_tool(query: str) -> Dict[str, Any]
def analyze_query_characteristics_tool(query: str) -> Dict[str, Any]
```

### 2. **Agentic Vector RAG Agent**

**Role**: Semantic document retrieval specialist

**Advanced Features**:
- **Adaptive Search Strategy**: Automatically selects optimal search approach
  - `auto`: Hybrid vector + BM25
  - `vector_only`: Pure semantic search
  - `bm25_only`: Keyword-based search
- **Dynamic Parameter Tuning**: Adjusts search parameters based on query type
- **Result Reranking**: LLM-powered relevance scoring
- **Learning Integration**: Improves search strategy based on success patterns

**Search Strategies**:
```python
# Strategy Selection Logic
if query_type == "conceptual":
    strategy = "vector_only"
elif query_type == "specific_term":
    strategy = "bm25_only"
else:
    strategy = "auto"  # Hybrid approach
```

**Performance Metrics**:
- Success rate tracking per strategy
- Average relevance scoring
- Response time optimization

### 3. **Agentic Graph RAG Agent**

**Role**: Knowledge graph navigation specialist

**Graph Operations**:
- **Entity Extraction**: Identifies medical entities from queries
- **Dynamic Cypher Generation**: Creates optimized Neo4j queries
- **Relationship Analysis**: Discovers entity relationships and patterns
- **Constraint Application**: Applies filters and constraints intelligently

**Query Examples**:
```cypher
// Patient demographic queries
MATCH (p:Patient {age: 17, gender: 'Male'})
WHERE p.finding_labels CONTAINS 'effusion'
RETURN COUNT(p)

// Disease correlation analysis
MATCH (p:Patient)-[:HAS_FINDING]->(f:Finding)
WHERE f.name = 'pneumonia'
RETURN p.age_group, COUNT(p) as case_count
```

### 4. **Validator Agent**

**Role**: Quality assurance and consistency checking

**Validation Dimensions**:
- **Content Accuracy**: Verifies factual correctness
- **Source Consistency**: Checks consistency across multiple sources
- **Medical Validity**: Ensures medically sound information
- **Completeness**: Assesses answer completeness

**Validation Process**:
```python
def validate_results(results: Dict[str, Any]) -> ValidationResult:
    # 1. Check document count and quality
    # 2. Assess content relevance
    # 3. Verify medical accuracy
    # 4. Calculate confidence scores
```

### 5. **Answer Synthesis Agent**

**Role**: Final answer generation and presentation

**Synthesis Features**:
- **Multi-Source Integration**: Combines vector and graph results seamlessly
- **Citation Management**: Tracks and formats source citations
- **Confidence Scoring**: Provides transparency in answer confidence
- **Medical Context**: Maintains medical accuracy and appropriate terminology

**Synthesis Pipeline**:
```python
def enhanced_synthesis(context: str, query: str) -> Dict[str, Any]:
    # 1. Analyze context relevance
    # 2. Generate comprehensive answer
    # 3. Add appropriate citations
    # 4. Calculate confidence score
```

## 🔄 Workflow Execution Patterns

### **Pattern 1: Document Retrieval Flow (Vector Search)**
```mermaid
graph LR
    A[📝 Query:<br/>'What is NIH Chest X-ray?'] --> B[🔍 Medical Validation<br/>LLM Check]
    B --> C[📊 Intent: Document<br/>Complexity: Simple]
    C --> D[🎯 Route: Vector Only]
    D --> E[🧠 Strategy Selection<br/>BM25 for specific terms]
    E --> F[🔍 Qdrant Search<br/>medical_research_doc]
    F --> G[📋 5 Documents Retrieved<br/>Relevance > 0.6]
    G --> H[✅ Validation<br/>Score: 0.8]
    H --> I[📝 Synthesis<br/>Confidence: 0.95]
    I --> J[📄 Final Answer<br/>+ Citations]
    
    style A fill:#e3f2fd
    style J fill:#e8f5e8
    style H fill:#fff3e0
```

### **Pattern 2: Relational Query Flow (Graph Search)**
```mermaid
graph LR
    A[📊 Query:<br/>'Male patients age 17<br/>with effusion?'] --> B[🔍 Medical Validation<br/>LLM Check]
    B --> C[📈 Intent: Relational<br/>Complexity: Moderate]
    C --> D[🎯 Route: Graph Only]
    D --> E[🏷️ Entity Extraction<br/>age=17, gender=Male<br/>finding=effusion]
    E --> F[⚡ Cypher Generation<br/>MATCH (p:Patient)<br/>WHERE conditions]
    F --> G[🕸️ Neo4j Query<br/>Patient relationships]
    G --> H[📊 Count Results<br/>Aggregated data]
    H --> I[✅ Validation<br/>Data consistency]
    I --> J[📝 Synthesis<br/>Statistical summary]
    J --> K[📄 Final Answer<br/>+ Data sources]
    
    style A fill:#e3f2fd
    style K fill:#e8f5e8
    style I fill:#fff3e0
```

### **Pattern 3: Hybrid Execution Flow (Sequential Both)**
```mermaid
graph TD
    A[🤔 Query:<br/>'Pneumonia cases and<br/>treatment protocols?'] --> B[🔍 Medical Validation]
    B --> C[🔀 Intent: Hybrid<br/>Complexity: High]
    C --> D[🎯 Route: Both Sequential]
    
    D --> E[📚 Vector Search Phase]
    E --> F[🔍 Qdrant: Treatment<br/>protocol documents]
    F --> G[📋 Document Results]
    
    D --> H[🕸️ Graph Search Phase]
    H --> I[🏷️ Extract: pneumonia<br/>patient entities]
    I --> J[⚡ Neo4j: Patient<br/>case relationships]
    J --> K[📊 Statistical Results]
    
    G --> L[🔄 Result Merging]
    K --> L
    L --> M[✅ Cross-Validation<br/>Documents + Statistics]
    M --> N[📝 Comprehensive Synthesis<br/>Protocols + Case data]
    N --> O[📄 Final Answer<br/>+ Multiple sources]
    
    style A fill:#e3f2fd
    style O fill:#e8f5e8
    style M fill:#fff3e0
```

### **Pattern 4: Direct LLM Response Flow (No Retrieval)**
```mermaid
graph LR
    A[❓ Query:<br/>'What is medical imaging?'] --> B[🔍 Medical Validation<br/>LLM Check]
    B --> C[🧠 Intent: General<br/>Knowledge available]
    C --> D[🎯 Route: Direct LLM]
    D --> E[🤖 Azure OpenAI<br/>Direct reasoning]
    E --> F[✅ Validation<br/>Medical accuracy]
    F --> G[📝 Synthesis<br/>General knowledge]
    G --> H[📄 Final Answer<br/>No external sources]
    
    style A fill:#e3f2fd
    style H fill:#e8f5e8
    style F fill:#fff3e0
```

### **Advanced Multi-Agent Coordination Flow**
```mermaid
stateDiagram-v2
    [*] --> QueryReceived
    QueryReceived --> MedicalValidation: User Input
    
    MedicalValidation --> QueryAnalysis: Valid Medical Query
    MedicalValidation --> Rejected: Invalid Query
    
    QueryAnalysis --> ToolSelection: Analysis Complete
    ToolSelection --> ExecutionPlanning: Tools Selected
    
    ExecutionPlanning --> VectorRoute: Document Intent
    ExecutionPlanning --> GraphRoute: Relational Intent
    ExecutionPlanning --> HybridRoute: Complex Intent
    ExecutionPlanning --> DirectRoute: General Intent
    
    VectorRoute --> VectorAgent: Execute
    GraphRoute --> GraphAgent: Execute
    HybridRoute --> VectorAgent: Phase 1
    DirectRoute --> DirectLLM: Execute
    
    VectorAgent --> ResultValidation: Vector Results
    GraphAgent --> ResultValidation: Graph Results
    HybridRoute --> GraphAgent: Phase 2
    DirectLLM --> ResultValidation: Direct Results
    
    ResultValidation --> AnswerSynthesis: Validated
    ResultValidation --> Fallback: Failed Validation
    
    AnswerSynthesis --> [*]: Final Answer
    Fallback --> ToolSelection: Retry Different Strategy
    Rejected --> [*]: Error Response
    
    note right of MedicalValidation
        LLM-powered validation
        Medical domain check
        Query relevance scoring
    end note
    
    note right of ResultValidation
        Multi-source consistency
        Quality assessment
        Confidence scoring
    end note
```

### **🗂️ Data Architecture & Storage Flow**

```mermaid
graph TB
    subgraph "📥 Data Ingestion Pipeline"
        PDF[📄 PDF Documents<br/>Medical Research Papers]
        CSV[📊 CSV Files<br/>Patient Data, Findings]
        BBOX[📋 BBox Data<br/>Medical Annotations]
        
        PDF --> PROC[📝 Document Processing<br/>Text extraction, Chunking]
        CSV --> PROC
        BBOX --> PROC
        
        PROC --> EMB[🧮 Embedding Generation<br/>Azure OpenAI Embeddings]
        PROC --> ENT[🏷️ Entity Extraction<br/>Medical entities, Relations]
        
        EMB --> QDRANT_STORE[(🔍 Qdrant Vector Store<br/>Collection: medical_research_doc<br/>- Document embeddings<br/>- Metadata indexing<br/>- Similarity search)]
        
        ENT --> NEO4J_STORE[(🕸️ Neo4j Graph Store<br/>- Patient nodes<br/>- Finding relationships<br/>- Medical entity connections)]
    end
    
    subgraph "🔄 Query Processing Flow"
        QUERY[❓ User Query] --> ROUTE{🎯 Routing Decision}
        
        ROUTE -->|Document Query| VEC_SEARCH[🔍 Vector Search]
        ROUTE -->|Relational Query| GRAPH_SEARCH[🕸️ Graph Search]
        ROUTE -->|Complex Query| HYBRID_SEARCH[🔀 Hybrid Search]
        
        VEC_SEARCH --> QDRANT_STORE
        GRAPH_SEARCH --> NEO4J_STORE
        HYBRID_SEARCH --> QDRANT_STORE
        HYBRID_SEARCH --> NEO4J_STORE
        
        QDRANT_STORE --> RESULTS[📋 Retrieved Results]
        NEO4J_STORE --> RESULTS
        
        RESULTS --> VALIDATE[✅ Validation & Synthesis]
        VALIDATE --> RESPONSE[📄 Final Response]
    end
    
    subgraph "☁️ External Services Integration"
        AZURE[☁️ Azure OpenAI<br/>- GPT-4o-mini<br/>- Embedding generation<br/>- LLM reasoning]
        KEYVAULT[🔐 Azure Key Vault<br/>- API keys<br/>- Connection strings<br/>- Security credentials]
        LANGSMITH[📊 LangSmith<br/>- Tracing<br/>- Monitoring<br/>- Debug analytics]
    end
    
    AZURE -.-> EMB
    AZURE -.-> VALIDATE
    KEYVAULT -.-> AZURE
    KEYVAULT -.-> QDRANT_STORE
    KEYVAULT -.-> NEO4J_STORE
    LANGSMITH -.-> ROUTE
    LANGSMITH -.-> VALIDATE
    
    %% Styling
    classDef ingestLayer fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef queryLayer fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef serviceLayer fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef storageLayer fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class PDF,CSV,BBOX,PROC,EMB,ENT ingestLayer
    class QUERY,ROUTE,VEC_SEARCH,GRAPH_SEARCH,HYBRID_SEARCH,RESULTS,VALIDATE,RESPONSE queryLayer
    class AZURE,KEYVAULT,LANGSMITH serviceLayer
    class QDRANT_STORE,NEO4J_STORE storageLayer
```

### **🏗️ Agent Collaboration Architecture**

```mermaid
graph TD
    subgraph "🎭 Orchestrator Agent Hub"
        ORCH_CORE[🧠 Core Orchestration<br/>- Query routing decisions<br/>- Agent coordination<br/>- Learning memory]
        ORCH_MED[🏥 Medical Validation<br/>- Domain relevance check<br/>- Medical accuracy verification]
        ORCH_LEARN[📚 Learning System<br/>- Performance tracking<br/>- Strategy optimization<br/>- Failure analysis]
    end
    
    subgraph "🔍 Retrieval Agent Specialists"
        VEC_AGENT[📚 Vector RAG Agent<br/>- Qdrant integration<br/>- Semantic search<br/>- BM25 hybrid search<br/>- Result ranking]
        
        GRAPH_AGENT[🕸️ Graph RAG Agent<br/>- Neo4j queries<br/>- Entity extraction<br/>- Relationship analysis<br/>- Cypher generation]
        
        VEC_STRATEGY[🧠 Vector Strategy<br/>- Auto: Hybrid approach<br/>- Vector: Semantic only<br/>- BM25: Keyword only]
        
        GRAPH_STRATEGY[🧠 Graph Strategy<br/>- Entity-focused queries<br/>- Relationship traversal<br/>- Aggregation operations]
    end
    
    subgraph "✅ Quality Assurance Agents"
        VALIDATOR[✅ Validator Agent<br/>- Result consistency<br/>- Medical accuracy<br/>- Source validation<br/>- Confidence scoring]
        
        SYNTHESIZER[📝 Synthesis Agent<br/>- Multi-source integration<br/>- Citation formatting<br/>- Medical context preservation<br/>- Final answer generation]
    end
    
    subgraph "🔄 Communication Protocols"
        STATE_MGMT[📊 State Management<br/>- WorkflowState tracking<br/>- Agent result passing<br/>- Error propagation]
        
        TOOL_REGISTRY[🛠️ Tool Registry<br/>- Available tool catalog<br/>- Dynamic tool selection<br/>- Performance metrics]
    end
    
    %% Agent Interactions
    ORCH_CORE --> VEC_AGENT
    ORCH_CORE --> GRAPH_AGENT
    ORCH_MED --> ORCH_CORE
    ORCH_LEARN --> ORCH_CORE
    
    VEC_AGENT --> VEC_STRATEGY
    GRAPH_AGENT --> GRAPH_STRATEGY
    
    VEC_AGENT --> VALIDATOR
    GRAPH_AGENT --> VALIDATOR
    VALIDATOR --> SYNTHESIZER
    
    STATE_MGMT --> ORCH_CORE
    STATE_MGMT --> VALIDATOR
    STATE_MGMT --> SYNTHESIZER
    
    TOOL_REGISTRY --> ORCH_CORE
    TOOL_REGISTRY --> VEC_AGENT
    TOOL_REGISTRY --> GRAPH_AGENT
    
    %% Feedback Loops
    SYNTHESIZER -.-> ORCH_LEARN
    VALIDATOR -.-> ORCH_LEARN
    VEC_AGENT -.-> ORCH_LEARN
    GRAPH_AGENT -.-> ORCH_LEARN
    
    %% Styling
    classDef orchestrator fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef retrieval fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef quality fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef communication fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class ORCH_CORE,ORCH_MED,ORCH_LEARN orchestrator
    class VEC_AGENT,GRAPH_AGENT,VEC_STRATEGY,GRAPH_STRATEGY retrieval
    class VALIDATOR,SYNTHESIZER quality
    class STATE_MGMT,TOOL_REGISTRY communication
```

## 🛡️ Security & Configuration

### Azure Key Vault Integration

**Secure Secret Management**:
```python
# Configuration retrieval
azure_endpoint = get_secret_from_keyvault("AZURE-OPENAI-ENDPOINT")
azure_api_key = get_secret_from_keyvault("AZURE-OPENAI-API-KEY")
qdrant_url = get_secret_from_keyvault("QDRANT-API-URL")
neo4j_uri = get_secret_from_keyvault("NEO4J-URI")
```

**Environment Configuration**:
- **Production**: Uses Azure Key Vault for all secrets
- **Development**: Falls back to `.env.dev` for local development
- **Testing**: Supports mock configurations for unit testing

### Input Sanitization

**Security Measures**:
- Prompt injection detection
- Input length validation
- Medical domain validation
- Output sanitization

## 📊 Performance & Monitoring

### Learning Memory System

**Adaptive Intelligence**:
- **Query Pattern Learning**: Improves routing decisions over time
- **Search Strategy Optimization**: Adapts search parameters based on success
- **Performance Tracking**: Monitors response times and accuracy
- **Failure Analysis**: Learns from failed queries to improve robustness

### Observability Features

**Monitoring Capabilities**:
- **LangSmith Integration**: Detailed tracing and debugging
- **Structured Logging**: Comprehensive activity logging
- **Performance Metrics**: Response time and accuracy tracking
- **Health Monitoring**: System component status tracking

## 🚀 Usage Examples

### Basic Query Processing

```python
from updated_agents.simple_agentic_app import EnhancedAgenticRAGApplication

# Initialize application
app = EnhancedAgenticRAGApplication()
result = app.initialize_system()

# Process query
query = "What is NIH Chest X-ray?"
response = app.process_query(query)

print(response['final_answer'])
```

### Advanced Configuration

```python
# Custom configuration with Key Vault
import os
os.environ['Keyvalue_Enabled'] = 'true'

# Initialize with specific parameters
app = EnhancedAgenticRAGApplication()
app.initialize_system()

# Process complex relational query
query = "Total number of male patients aged 17 with effusion?"
response = app.process_query(query)
```

### Streamlit Web Interface

```python
# Run the web interface
streamlit run src/updated_agents/simple_agentic_streamlit.py

# Access at: http://localhost:8501
```

## 🔧 Configuration

### Required Environment Variables

**Azure OpenAI Configuration**:
```env
AZURE-OPENAI-ENDPOINT=https://your-openai.openai.azure.com/
AZURE-OPENAI-API-KEY=your-api-key
AZURE-OPENAI-DEPLOYMENT=gpt-4o-mini
AZURE-OPENAI-API-VERSION=2024-02-15-preview
```

**Vector Database Configuration**:
```env
QDRANT-API-URL=https://your-cluster.qdrant.io:6333
QDRANT-API-KEY-VAL=your-qdrant-api-key
QDRANT-COLLECTION=medical_research_doc
```

**Graph Database Configuration**:
```env
NEO4J-URI=bolt://localhost:7687
NEO4J-USERNAME=neo4j
NEO4J-PASSWORD=your-password
```

### Azure Key Vault Setup

1. Create Azure Key Vault resource
2. Add secrets with exact names matching environment variables
3. Configure Azure CLI authentication
4. Set `Keyvalue_Enabled=true` in environment

## 🎯 Key Benefits

### 1. **True Agentic Behavior**
- Each agent has autonomous reasoning capabilities
- Learning memory system for continuous improvement
- Adaptive decision-making based on context

### 2. **Production-Ready Architecture**
- Comprehensive error handling and recovery
- Security-first design with input sanitization
- Scalable component architecture

### 3. **Multi-Modal Intelligence**
- Vector search for semantic understanding
- Graph search for relational queries
- Hybrid approaches for complex scenarios

### 4. **Medical Domain Expertise**
- Specialized medical validation
- Domain-specific entity recognition
- Medically accurate response generation

### 5. **Enterprise Integration**
- Azure Key Vault for secure credential management
- LangSmith for observability and debugging
- Streamlit for user-friendly interface

## 🔮 Future Enhancements

### Planned Features
- **Multi-Language Support**: Extend to non-English medical queries
- **Advanced Reasoning**: Integration with reasoning frameworks
- **Custom Agent Development**: Framework for domain-specific agents
- **Advanced Analytics**: Enhanced performance analytics and insights
- **Federated Learning**: Distributed learning across deployments

### Extensibility Points
- **New Agent Types**: Easy addition of specialized agents
- **Additional Databases**: Support for additional vector/graph databases
- **Custom Tools**: Framework for custom tool development
- **Workflow Customization**: Flexible workflow modification capabilities

---

## 📚 Related Documentation

- [Architecture Overview](architecture.md)
- [Installation Guide](../README.md)
- [API Reference](api_reference.md)
- [Deployment Guide](deployment.md)

---

**Version**: 1.0.0  
**Last Updated**: August 21, 2025  
**Author**: Enhanced Agentic RAG Development Team
