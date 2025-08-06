from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from multi_agent_rag_workflow import MultiAgentRAGWorkflow
from logging_config import get_logger

# Configure the API
app = FastAPI(
    title="Multi-Agent RAG API",
    description="Advanced RAG system with Vector and Graph retrieval, validation, and synthesis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging - logging will be configured by the workflow
logger = get_logger("api")

# Global workflow instance
workflow_instance = None


class QueryRequest(BaseModel):
    """Request model for queries"""
    query: str
    session_id: Optional[str] = None
    include_metadata: Optional[bool] = True


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str
    citations: list
    confidence_score: float
    route_taken: str
    validation_passed: bool
    status: str
    metrics: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the workflow on startup"""
    global workflow_instance
    try:
        workflow_instance = MultiAgentRAGWorkflow()
        logger.info("api_startup_completed")
    except Exception as e:
        logger.error("api_startup_failed", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global workflow_instance
    if workflow_instance:
        workflow_instance.close()
        logger.info("api_shutdown_completed")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Agent RAG Workflow API",
        "version": "1.0.0",
        "description": "Advanced RAG system with orchestration, vector search, graph queries, validation, and synthesis",
        "endpoints": {
            "query": "/query - POST endpoint for processing queries",
            "health": "/health - Health check endpoint",
            "metrics": "/metrics - Prometheus metrics endpoint"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query through the multi-agent RAG workflow
    
    This endpoint implements the complete flowchart:
    1. Orchestrator routes the query
    2. Vector-RAG or Graph-RAG retrieves information
    3. Validator checks consistency
    4. Answer Synthesis composes the final response
    """
    global workflow_instance
    
    if not workflow_instance:
        raise HTTPException(status_code=503, detail="Workflow not initialized")
    
    try:
        logger.info(
            "api_query_received",
            query_length=len(request.query),
            session_id=request.session_id
        )
        
        # Process the query
        response = workflow_instance.process_query(
            query=request.query,
            session_id=request.session_id
        )
        
        # Include or exclude metadata based on request
        if not request.include_metadata:
            response["metadata"] = None
        
        logger.info(
            "api_query_completed",
            session_id=request.session_id,
            status=response["status"],
            route=response["route_taken"]
        )
        
        return QueryResponse(**response)
        
    except Exception as e:
        logger.error("api_query_error", error=str(e), session_id=request.session_id)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global workflow_instance
    
    status = "healthy" if workflow_instance else "unhealthy"
    
    return {
        "status": status,
        "timestamp": "2025-08-01T00:00:00Z",
        "components": {
            "workflow": status,
            "llm": "healthy" if workflow_instance and hasattr(workflow_instance, 'llm') else "unhealthy",
            "vector_store": "healthy" if workflow_instance and hasattr(workflow_instance, 'qdrant_client') else "unhealthy",
            "graph_store": "healthy" if workflow_instance and hasattr(workflow_instance, 'neo4j_driver') else "unhealthy"
        }
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return JSONResponse(
        content=generate_latest().decode('utf-8'),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/workflow/info")
async def workflow_info():
    """Get information about the workflow structure"""
    return {
        "workflow_type": "Multi-Agent RAG",
        "agents": [
            {
                "name": "Orchestrator",
                "role": "Query routing and decision making",
                "reads": ["state.query"],
                "writes": ["state.route", "state.latency_ms['orch']"]
            },
            {
                "name": "Vector-RAG",
                "role": "Semantic similarity search using Qdrant",
                "reads": ["state.query"],
                "writes": ["state.vector_docs", "state.latency_ms['vec']", "state.memory_usage['vec']"],
                "tools": ["Qdrant search", "Optional LLM re-ranking"]
            },
            {
                "name": "Graph-RAG",
                "role": "Entity extraction and knowledge graph queries using Neo4j",
                "reads": ["state.query"],
                "writes": ["state.graph_triples", "state.latency_ms['graph']", "state.memory_usage['graph']"],
                "tools": ["Neo4j Cypher", "Entity extraction"]
            },
            {
                "name": "Validator",
                "role": "Consistency validation between retrieval sources",
                "reads": ["state.vector_docs", "state.graph_triples"],
                "writes": ["state.validation_passed", "state.validation_errors", "state.latency_ms['val']"],
                "methods": ["LLM validation", "Rule-based validation"]
            },
            {
                "name": "Answer Synthesis",
                "role": "Final answer composition with citations",
                "reads": ["state.vector_docs", "state.graph_triples", "state.validation_passed"],
                "writes": ["state.answer", "state.citations", "state.latency_ms['ans']", "state.memory_usage['ans']"],
                "capabilities": ["LLM composition", "Citation generation", "Confidence scoring"]
            }
        ],
        "observability": {
            "logging": "Structured logging with trace IDs",
            "metrics": "Prometheus metrics for latency, memory, errors",
            "tracing": "Full request tracing through all agents"
        },
        "routing_options": ["vector", "graph", "both", "none"]
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
