from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass, field
import time
from datetime import datetime


@dataclass
class ValidationResult:
    """Results from validation process"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    consistency_issues: List[str] = field(default_factory=list)


class WorkflowState(TypedDict):
    """State for multi-agent RAG workflow - Happy Path focus"""
    
    # Input
    query: str
    session_id: Optional[str]
    
    # Routing decision
    route: Optional[str]  # "vector", "graph", "both", "none"
    
    # Retrieved data from different sources
    vector_docs: Optional[List[Dict[str, Any]]]
    graph_triples: Optional[List[Dict[str, Any]]]
    
    # Validation results
    validation_passed: Optional[bool]
    validation_errors: Optional[List[str]]
    validation_result: Optional[ValidationResult]
    
    # Final synthesized output
    answer: Optional[str]
    
    # Performance metrics per agent
    latency_ms: Dict[str, float]  # {"orch": 150, "vec": 300, "graph": 450, "val": 100, "ans": 200}
    memory_usage: Dict[str, float]  # Memory usage in MB per agent
    
    # Metadata and tracking
    timestamp: Optional[str]
    processing_start: Optional[float]
    
    # Error handling
    errors: Optional[List[str]]
    status: Optional[str]  # "processing", "completed", "failed", "partial"
    
    # Observability data
    trace_id: Optional[str]
    metrics: Optional[Dict[str, Any]]


def create_initial_state(query: str, session_id: str = None) -> WorkflowState:
    """
    Create initial workflow state for Happy Path flow
    
    Initializes all fields needed for the workflow execution
    """
    return WorkflowState(
        query=query,
        session_id=session_id or f"session_{int(time.time())}",
        route=None,
        vector_docs=None,
        graph_triples=None,
        validation_passed=None,
        validation_errors=None,
        validation_result=None,
        answer=None,
        latency_ms={},
        memory_usage={},
        timestamp=datetime.now().isoformat(),
        processing_start=time.time(),
        errors=[],
        status="processing",
        trace_id=f"trace_{int(time.time())}_{hash(query) % 10000}",
        metrics={}
    )
