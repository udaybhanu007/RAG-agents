"""
LangGraph-based Agentic Workflow

This module provides a LangGraph-compliant implementation of the existing 
agentic workflow while preserving all business logic and agentic behaviors.

Key Features:
- Minimal changes to existing business logic
- Standard LangGraph StateGraph with nodes and edges
- Preserves all agentic capabilities
- Maintains state-based inter-agent communication
- Keeps existing security and observability features
"""

import os
import sys
from typing import Dict, Any, Literal
from typing_extensions import TypedDict

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI

# Import existing agents and base classes
from .simple_agentic_agents import (
    AgenticOrchestratorAgent,
    AgenticVectorRAGAgent, 
    AgenticGraphRAGAgent,
    SimpleValidatorAgent,
    SimpleAnswerSynthesisAgent
)
from .base_classes import WorkflowState as BaseWorkflowState
from core.logging_config import get_logger

logger = get_logger("langgraph_agentic_workflow")

# LangGraph State Schema - extending existing WorkflowState
class LangGraphAgenticState(TypedDict):
    """LangGraph-compliant state schema that preserves existing state structure"""
    # Core query data
    query: str
    trace_id: str
    
    # Medical validation
    medical_validation: Dict[str, Any]
    
    # Orchestrator reasoning and planning
    reasoning_plan: Dict[str, Any]
    selected_route: str
    routing_decision: str
    
    # Agent results
    vector_results: Dict[str, Any]
    graph_results: Dict[str, Any]
    
    # Validation and synthesis
    validation_result: Dict[str, Any]
    validated_results: list  # CRITICAL: Add validated_results to schema
    final_answer: str
    sources: list
    confidence_score: float
    
    # Agentic metadata
    agentic_indicators: Dict[str, Any]
    learning_update: Dict[str, Any]
    execution_metrics: Dict[str, Any]
    
    # Processing metadata
    current_step: str
    is_complete: bool
    errors: list

class LangGraphAgenticWorkflow:
    """
    LangGraph-compliant agentic workflow that preserves all existing business logic
    
    This class wraps the existing agentic agents in a standard LangGraph StateGraph
    while maintaining all the sophisticated agentic behaviors, learning capabilities,
    and security features.
    """
    
    def __init__(self, llm: AzureChatOpenAI, vector_store, graph_store):
        """Initialize with existing agents - no changes to business logic"""
        logger.info("initializing_langgraph_agentic_workflow")
        
        # Initialize all existing agents with their full capabilities
        self.vector_agent = AgenticVectorRAGAgent(llm, vector_store)
        self.graph_agent = AgenticGraphRAGAgent(llm, graph_store)
        self.orchestrator = AgenticOrchestratorAgent(llm, self.vector_agent, self.graph_agent)
        self.validator = SimpleValidatorAgent(llm)
        self.synthesizer = SimpleAnswerSynthesisAgent(llm)
        
        # Build the LangGraph workflow
        self.workflow = self._build_langgraph_workflow()
        self.execution_count = 0
        
        logger.info("langgraph_agentic_workflow_initialized")
    
    def _build_langgraph_workflow(self) -> StateGraph:
        """Build the LangGraph StateGraph with existing agent logic"""
        logger.info("building_langgraph_workflow")
        
        # Create LangGraph StateGraph
        workflow = StateGraph(LangGraphAgenticState)
        
        # Add nodes that wrap existing agent methods
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("vector_search", self._vector_search_node)
        workflow.add_node("graph_search", self._graph_search_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Add conditional routing based on orchestrator decisions
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_after_orchestrator,
            {
                "vector": "vector_search",
                "graph": "graph_search", 
                "both": "vector_search",
                "none": END
            }
        )
        
        # Handle "both" route continuation
        workflow.add_conditional_edges(
            "vector_search",
            self._route_after_vector,
            {
                "continue_to_graph": "graph_search",
                "continue_to_validator": "validator"
            }
        )
        
        # Simple edges
        workflow.add_edge("graph_search", "validator")
        workflow.add_edge("validator", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        # Compile the workflow
        compiled_workflow = workflow.compile()
        
        logger.info("langgraph_workflow_compiled")
        return compiled_workflow
    
    def _orchestrator_node(self, state: LangGraphAgenticState) -> LangGraphAgenticState:
        """Orchestrator node - wraps existing orchestrator logic"""
        logger.info("orchestrator_node_started", trace_id=state.get("trace_id"))
        
        # Convert LangGraph state to WorkflowState for existing agents
        workflow_state = self._convert_to_workflow_state(state)
        
        # Use existing orchestrator logic with full agentic capabilities
        result_state = self.orchestrator.reason_and_plan(workflow_state)
        
        # Update LangGraph state with results
        state.update({
            "medical_validation": result_state.get("medical_validation", {}),
            "reasoning_plan": result_state.get("reasoning_plan", {}),
            "selected_route": result_state.get("reasoning_plan", {}).get("selected_route", "none"),
            "routing_decision": result_state.get("reasoning_plan", {}).get("reasoning", ""),
            "vector_results": result_state.get("vector_results", {}),
            "graph_results": result_state.get("graph_results", {}),
            "current_step": "orchestrator_complete"
        })
        
        logger.info("orchestrator_node_completed", 
                   route=state["selected_route"],
                   trace_id=state.get("trace_id"))
        return state
    
    def _vector_search_node(self, state: LangGraphAgenticState) -> LangGraphAgenticState:
        """Vector search node - wraps existing vector agent logic"""
        logger.info("vector_search_node_started", trace_id=state.get("trace_id"))
        
        # Only execute if not already done by orchestrator
        if not state.get("vector_results", {}).get("documents"):
            workflow_state = self._convert_to_workflow_state(state)
            result_state = self.vector_agent.search_with_optimization(workflow_state)
            
            state["vector_results"] = result_state.get("vector_results", {})
        
        state["current_step"] = "vector_search_complete"
        logger.info("vector_search_node_completed", 
                   docs_found=len(state["vector_results"].get("documents", [])),
                   trace_id=state.get("trace_id"))
        return state
    
    def _graph_search_node(self, state: LangGraphAgenticState) -> LangGraphAgenticState:
        """Graph search node - wraps existing graph agent logic"""
        logger.info("graph_search_node_started", trace_id=state.get("trace_id"))
        
        # Only execute if not already done by orchestrator
        if not state.get("graph_results", {}).get("documents"):
            workflow_state = self._convert_to_workflow_state(state)
            result_state = self.graph_agent.search_with_optimization(workflow_state)
            
            state["graph_results"] = result_state.get("graph_results", {})
        
        state["current_step"] = "graph_search_complete"
        logger.info("graph_search_node_completed",
                   docs_found=len(state["graph_results"].get("documents", [])),
                   trace_id=state.get("trace_id"))
        return state
    
    def _validator_node(self, state: LangGraphAgenticState) -> LangGraphAgenticState:
        """Validator node - wraps existing validator logic"""
        logger.info("validator_node_started", trace_id=state.get("trace_id"))
        
        workflow_state = self._convert_to_workflow_state(state)
        validation_result = self.validator.validate_results(workflow_state)
        
        # CRITICAL FIX: Preserve validated_results from validator
        validated_results = workflow_state.get("validated_results", [])
        
        state.update({
            "validation_result": {
                "is_valid": validation_result.is_valid,
                "score": validation_result.score,
                "feedback": validation_result.feedback
            },
            "validated_results": validated_results,  # Preserve validated_results
            "current_step": "validation_complete"
        })
        
        logger.info("validator_node_completed",
                   is_valid=validation_result.is_valid,
                   validated_results_count=len(validated_results),
                   trace_id=state.get("trace_id"))
        return state
    
    def _synthesizer_node(self, state: LangGraphAgenticState) -> LangGraphAgenticState:
        """Synthesizer node - wraps existing synthesizer logic"""
        logger.info("synthesizer_node_started", trace_id=state.get("trace_id"))
        
        workflow_state = self._convert_to_workflow_state(state)
        synthesis_result = self.synthesizer.synthesize_answer(workflow_state)
        
        state.update({
            "final_answer": synthesis_result.answer,
            "sources": synthesis_result.sources,
            "confidence_score": synthesis_result.confidence,
            "current_step": "synthesis_complete",
            "is_complete": True
        })
        
        logger.info("synthesizer_node_completed",
                   answer_length=len(synthesis_result.answer),
                   confidence=synthesis_result.confidence,
                   trace_id=state.get("trace_id"))
        return state
    
    def _route_after_orchestrator(self, state: LangGraphAgenticState) -> Literal["vector", "graph", "both", "none"]:
        """Route after orchestrator based on its decision"""
        route = state.get("selected_route", "none")
        
        # Non-medical queries go straight to END
        medical_validation = state.get("medical_validation", {})
        if not medical_validation.get("is_medical", False):
            return "none"
        
        logger.debug("routing_after_orchestrator", route=route)
        return route
    
    def _route_after_vector(self, state: LangGraphAgenticState) -> Literal["continue_to_graph", "continue_to_validator"]:
        """Route after vector search based on original plan"""
        route = state.get("selected_route", "vector")
        
        if route == "both":
            return "continue_to_graph"
        else:
            return "continue_to_validator"
    
    def _convert_to_workflow_state(self, langgraph_state: LangGraphAgenticState) -> BaseWorkflowState:
        """Convert LangGraph state to existing WorkflowState format"""
        workflow_state = BaseWorkflowState()
        
        # Copy all data from LangGraph state
        for key, value in langgraph_state.items():
            workflow_state[key] = value
        
        return workflow_state
    
    def _convert_from_workflow_state(self, workflow_state: BaseWorkflowState) -> Dict[str, Any]:
        """Convert WorkflowState back to LangGraph state format"""
        return dict(workflow_state)
    
    def process_query(self, query: str, trace_id: str = None) -> Dict[str, Any]:
        """
        Process query using LangGraph workflow while preserving all existing capabilities
        
        This method maintains the same interface as the original SimpleAgenticWorkflow
        but uses LangGraph's StateGraph for execution.
        """
        from datetime import datetime
        import uuid
        
        if trace_id is None:
            trace_id = f"trace_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"
        
        logger.info("langgraph_query_processing_started", 
                   query_length=len(query),
                   trace_id=trace_id)
        
        start_time = datetime.now()
        self.execution_count += 1
        
        try:
            # Initialize LangGraph state
            initial_state: LangGraphAgenticState = {
                "query": query,
                "trace_id": trace_id,
                "medical_validation": {},
                "reasoning_plan": {},
                "selected_route": "",
                "routing_decision": "",
                "vector_results": {},
                "graph_results": {},
                "validation_result": {},
                "validated_results": [],  # Initialize validated_results
                "final_answer": "",
                "sources": [],
                "confidence_score": 0.0,
                "agentic_indicators": {},
                "learning_update": {},
                "execution_metrics": {},
                "current_step": "starting",
                "is_complete": False,
                "errors": []
            }
            
            # Execute the LangGraph workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare response in expected format
            result = {
                "answer": final_state.get("final_answer", "No answer generated"),
                "sources": final_state.get("sources", []),
                "confidence_score": final_state.get("confidence_score", 0.0),
                "reasoning_plan": final_state.get("reasoning_plan", {}),
                "learning_update": final_state.get("learning_update", {}),
                "execution_metrics": {
                    "execution_time": execution_time,
                    "execution_count": self.execution_count
                },
                "agentic_indicators": {
                    "autonomous_reasoning": bool(final_state.get("reasoning_plan")),
                    "learning_applied": bool(final_state.get("learning_update")),
                    "adaptive_behavior": True,
                    "langgraph_orchestrated": True  # New indicator
                }
            }
            
            logger.info("langgraph_query_processing_completed",
                       execution_time=execution_time,
                       has_answer=bool(result["answer"]),
                       confidence=result["confidence_score"],
                       trace_id=trace_id)
            
            return result
            
        except Exception as e:
            logger.error("langgraph_query_processing_failed", 
                        error=str(e), 
                        trace_id=trace_id)
            return {
                "answer": f"Processing failed: {str(e)}",
                "sources": [],
                "confidence_score": 0.0,
                "error": True,
                "agentic_indicators": {
                    "autonomous_reasoning": False,
                    "learning_applied": False,
                    "adaptive_behavior": False,
                    "langgraph_orchestrated": False
                }
            }
    
    def get_agentic_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights - delegates to existing implementation"""
        # Get insights from orchestrator and agents
        orchestrator_stats = self.orchestrator.get_learning_stats()
        
        insights = {
            "orchestrator": {
                "learning_stats": orchestrator_stats,
                "reasoning_history_length": len(self.orchestrator.reasoning_history),
                "available_tools": len(self.orchestrator.available_tools),
                "tool_performance": {
                    tool: info["success_rate"] 
                    for tool, info in self.orchestrator.available_tools.items()
                }
            },
            "vector_agent": {
                "search_strategies": self.vector_agent.strategy_performance,
                "search_history_length": len(self.vector_agent.search_history),
                "adaptive_params": self.vector_agent.adaptive_params
            },
            "graph_agent": {
                "relationship_patterns": len(self.graph_agent.relationship_patterns),
                "search_strategies": self.graph_agent.search_strategies,
                "optimization_count": self.graph_agent.query_optimizations,
                "adaptive_params": self.graph_agent.adaptive_params
            },
            "workflow": {
                "total_executions": self.execution_count,
                "framework": "LangGraph",  # New field
                "agentic_capabilities": {
                    "dynamic_reasoning": True,
                    "goal_oriented_planning": True,
                    "adaptive_learning": True,
                    "tool_orchestration": True,
                    "strategy_optimization": True,
                    "langgraph_state_management": True  # New capability
                }
            }
        }
        
        return insights

# Factory function for easy instantiation
def create_langgraph_agentic_workflow(llm: AzureChatOpenAI, vector_store, graph_store) -> LangGraphAgenticWorkflow:
    """Create a LangGraph-based agentic workflow with all required components"""
    logger.info("creating_langgraph_agentic_workflow")
    workflow = LangGraphAgenticWorkflow(llm, vector_store, graph_store)
    logger.info("langgraph_agentic_workflow_created")
    return workflow
