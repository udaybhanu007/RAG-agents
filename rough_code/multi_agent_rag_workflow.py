import os
from dotenv import load_dotenv
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from neo4j import GraphDatabase

# Import our custom modules
from workflow_state import WorkflowState, create_initial_state
from agents import OrchestratorAgent, VectorRAGAgent, GraphRAGAgent
from validation_synthesis import ValidatorAgent, AnswerSynthesisAgent
from observability import observability
from logging_config import configure_logging, get_logger

# Load environment variables
load_dotenv()

# Configure centralized logging once at startup
configure_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    enable_json=os.getenv("ENABLE_JSON_LOGS", "true").lower() == "true",
    enable_colors=os.getenv("ENABLE_COLORED_LOGS", "false").lower() == "true"
)

logger = get_logger("workflow_engine")


class MultiAgentRAGWorkflow:
    """
    Simple Multi-Agent RAG Workflow - Happy Path Implementation
    
    This implements the clean happy path flow:
    - Orchestrator Agent (routing)
    - Vector-RAG Agent (Qdrant search)
    - Graph-RAG Agent (Neo4j queries)
    - Validator Agent (consistency checking)
    - Answer Synthesis Agent (final composition)
    - Observability (metrics and logging)
    
    Focus: Clean, simple implementation without retry complexity
    """
    
    def __init__(self):
        self.initialize_components()
        self.build_workflow()
    
    def initialize_components(self):
        """Initialize all LLMs, databases, and agents"""
        try:
            # Initialize LLM
            self.llm = AzureChatOpenAI(
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                temperature=0.0
            )
            
            # Initialize embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
            )
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            
            # Initialize Neo4j driver
            self.neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                auth=(
                    os.getenv("NEO4J_USERNAME", "neo4j"),
                    os.getenv("NEO4J_PASSWORD", "password")
                )
            )
            
            # Initialize all agents
            self.orchestrator = OrchestratorAgent(self.llm)
            self.vector_rag = VectorRAGAgent(
                self.qdrant_client, 
                self.embeddings,
                collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
                llm=self.llm
            )
            self.graph_rag = GraphRAGAgent(self.neo4j_driver, self.llm)
            self.validator = ValidatorAgent(self.llm)
            self.synthesizer = AnswerSynthesisAgent(self.llm)
            
            logger.info("workflow_components_initialized")
            
        except Exception as e:
            logger.error("component_initialization_failed", error=str(e))
            raise
    
    def build_workflow(self):
        """
        Build the simple LangGraph workflow - Happy Path only
        
        Simple linear flow:
        1. Orchestrator decides routing
        2. Vector and/or Graph retrieval
        3. Validation
        4. Synthesis
        """
        
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add simple agent nodes
        workflow.add_node("orchestrator", self.orchestrator_node)
        workflow.add_node("vector_rag", self.vector_rag_node)
        workflow.add_node("graph_rag", self.graph_rag_node)
        workflow.add_node("validator", self.validator_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        workflow.add_node("no_data_response", self.no_data_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Simple routing from orchestrator
        workflow.add_conditional_edges(
            "orchestrator",
            self.route_query,
            {
                "vector": "vector_rag",
                "graph": "graph_rag", 
                "both_vector_first": "vector_rag",
                "none": "no_data_response"
            }
        )
        
        # For "both" routing, handle vector -> graph flow
        workflow.add_conditional_edges(
            "vector_rag",
            self.check_if_graph_needed,
            {
                "continue_to_graph": "graph_rag",
                "continue_to_validator": "validator"
            }
        )
        
        # Simple paths to validator
        workflow.add_edge("graph_rag", "validator")
        
        # Validator always goes to synthesizer (happy path)
        workflow.add_edge("validator", "synthesizer")
        
        # End points
        workflow.add_edge("synthesizer", END)
        workflow.add_edge("no_data_response", END)
        
        # Compile the workflow
        self.workflow = workflow.compile()
        
        logger.info("simple_workflow_compiled_happy_path_only")
    
    
    def orchestrator_node(self, state: WorkflowState) -> WorkflowState:
        """
        Orchestrator agent node - Simple routing decision
        Measures performance and logs routing decisions
        """
        with observability.measure_agent_performance("orch", state):
            result = self.orchestrator.route_query(state)
            # Log routing decision
            observability.log_routing_decision(result)
            return result
    
    def vector_rag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Vector RAG agent node - Simple semantic search
        Performs search using Qdrant vector database
        """
        with observability.measure_agent_performance("vec", state):
            result = self.vector_rag.retrieve_documents(state)
            # Log retrieval results
            observability.log_retrieval_results(result, "vector")
            return result
    
    def graph_rag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Graph RAG agent node - Simple graph queries
        Performs knowledge graph queries using Neo4j
        """
        with observability.measure_agent_performance("graph", state):
            result = self.graph_rag.extract_and_query(state)
            # Log retrieval results
            observability.log_retrieval_results(result, "graph")
            return result
    
    def validator_node(self, state: WorkflowState) -> WorkflowState:
        """
        Validator agent node - Simple consistency checking
        Happy path: validation should generally pass
        """
        with observability.measure_agent_performance("val", state):
            result = self.validator.validate_results(state)
            # Log validation results
            observability.log_validation_results("validation", result)
            return result
    
    def synthesizer_node(self, state: WorkflowState) -> WorkflowState:
        """
        Answer synthesis agent node - Final answer composition
        Creates coherent response with citations
        """
        with observability.measure_agent_performance("ans", state):
            result = self.synthesizer.synthesize_answer(state)
            # Log completion
            observability.log_query_completion(result)
            return result
    
    def no_data_node(self, state: WorkflowState) -> WorkflowState:
        """Handle queries that cannot be answered"""
        state["answer"] = "I don't have sufficient information to answer this query based on the available data sources."
        state["citations"] = []
        state["confidence_score"] = 0.0
        state["status"] = "completed"
        
        # Log completion
        observability.log_query_completion(state)
        
        return state
    
    def route_query(self, state: WorkflowState) -> str:
        """Route the query based on orchestrator decision"""
        route = state.get("route", "both")
        
        if route == "vector":
            return "vector"
        elif route == "graph":
            return "graph"
        elif route == "both":
            return "both_vector_first"  # Start with vector, then graph
        else:
            return "none"
    
    def check_if_graph_needed(self, state: WorkflowState) -> str:
        """
        Simple check if we need to continue to graph after vector retrieval
        """
        route = state.get("route", "")
        if route == "both":
            return "continue_to_graph"
        else:
            return "continue_to_validator"
    
    def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process a single query through the entire workflow with enhanced retry logic
        
        Args:
            query (str): The user's query
            session_id (str, optional): Session identifier for tracking
            
        Returns:
            Dict[str, Any]: Complete response with answer, citations, and metrics including retry context
        """
        try:
            # Create initial state with retry fields
            initial_state = create_initial_state(query, session_id)
            
            logger.info(
                "query_processing_started",
                query=query[:100],
                session_id=session_id,
                trace_id=initial_state["trace_id"]
            )
            
            # Run the workflow
            final_state = None
            for state in self.workflow.stream(initial_state):
                final_state = state
            
            # Extract the final state (LangGraph returns dict with node names as keys)
            if isinstance(final_state, dict):
                # Get the last state from the workflow execution
                for node_name, node_state in final_state.items():
                    final_state = node_state
                    break
            
            # Format response for Happy Path
            response = {
                "answer": final_state.get("answer", "No answer generated"),
                "citations": [
                    {
                        "type": citation.type,
                        "content": citation.content,
                        "score": citation.score,
                        "source_id": citation.source_id,
                        "metadata": citation.metadata
                    }
                    for citation in final_state.get("citations", [])
                ],
                "confidence_score": final_state.get("confidence_score", 0.0),
                "route_taken": final_state.get("route", "unknown"),
                "validation_passed": final_state.get("validation_passed", False),
                "status": final_state.get("status", "unknown"),
                "metrics": {
                    "latency_ms": final_state.get("latency_ms", {}),
                    "memory_usage": final_state.get("memory_usage", {}),
                    "total_latency_ms": sum(final_state.get("latency_ms", {}).values()),
                    "errors": final_state.get("errors", [])
                },
                "metadata": {
                    "session_id": final_state.get("session_id"),
                    "trace_id": final_state.get("trace_id"),
                    "timestamp": final_state.get("timestamp"),
                    "vector_docs_count": len(final_state.get("vector_docs", [])),
                    "graph_triples_count": len(final_state.get("graph_triples", []))
                }
            }
            
            logger.info(
                "query_processing_completed",
                session_id=session_id,
                trace_id=final_state.get("trace_id"),
                status=response["status"],
                total_latency=response["metrics"]["total_latency_ms"]
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "workflow_error",
                error=str(e),
                query=query[:100],
                session_id=session_id
            )
            
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "citations": [],
                "confidence_score": 0.0,
                "route_taken": "error",
                "validation_passed": False,
                "status": "failed",
                "metrics": {
                    "latency_ms": {},
                    "memory_usage": {},
                    "total_latency_ms": 0,
                    "errors": [str(e)]
                },
                "metadata": {
                    "session_id": session_id,
                    "trace_id": None,
                    "timestamp": None,
                    "vector_docs_count": 0,
                    "graph_triples_count": 0
                }
            }
    
    def close(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'neo4j_driver'):
                self.neo4j_driver.close()
            if hasattr(self, 'qdrant_client'):
                self.qdrant_client.close()
            logger.info("workflow_resources_closed")
        except Exception as e:
            logger.error("resource_cleanup_error", error=str(e))


# Example usage and testing functions
def create_sample_workflow():
    """Create a sample workflow instance"""
    return MultiAgentRAGWorkflow()


def test_workflow_with_sample_queries():
    """Test the workflow with sample queries"""
    workflow = create_sample_workflow()
    
    sample_queries = [
        "What is machine learning?",
        "How are neural networks related to deep learning?",
        "Explain the relationship between AI and robotics",
        "What are the applications of computer vision?"
    ]
    
    for query in sample_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        response = workflow.process_query(query)
        
        print(f"Answer: {response['answer']}")
        print(f"Route: {response['route_taken']}")
        print(f"Confidence: {response['confidence_score']}")
        print(f"Citations: {len(response['citations'])}")
        print(f"Total Latency: {response['metrics']['total_latency_ms']:.2f}ms")
    
    workflow.close()


if __name__ == "__main__":
    # Test the workflow
    test_workflow_with_sample_queries()
