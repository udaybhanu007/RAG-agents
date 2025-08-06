import os
from dotenv import load_dotenv
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
            self.orchestrator = OrchestratorAgent()  # No LLM needed
            self.vector_rag = VectorRAGAgent(
                self.qdrant_client, 
                self.embeddings,
                collection_name=os.getenv("QDRANT_COLLECTION", "documents"),
                llm=self.llm
            )
            self.graph_rag = GraphRAGAgent(self.neo4j_driver, self.llm)
            self.validator = ValidatorAgent()  # No LLM needed for rule-based validation
            self.synthesizer = AnswerSynthesisAgent(self.llm)
            
            logger.info("workflow_components_initialized")
            
        except Exception as e:
            logger.error("component_initialization_failed", error=str(e))
            raise
    
    def build_workflow(self):
        """
        Build the simple LangGraph workflow - Happy Path only
        
        Simple linear flow:
        1. Orchestrator decides routing (always returns vector/graph/both)
        2. Vector and/or Graph retrieval based on routing
        3. Validation
        4. Synthesis
        
        Note: No "none" routing case since OrchestratorAgent always returns valid routes
        """
        
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add simple agent nodes
        workflow.add_node("orchestrator", self.orchestrator_node)
        workflow.add_node("vector_rag", self.vector_rag_node)
        workflow.add_node("graph_rag", self.graph_rag_node)
        workflow.add_node("validator", self.validator_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Simple routing from orchestrator
        workflow.add_conditional_edges(
            "orchestrator",
            self.route_query,
            {
                "vector": "vector_rag",
                "graph": "graph_rag", 
                "both_vector_first": "vector_rag"
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
            return result
    
    def vector_rag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Vector RAG agent node - Simple semantic search
        Performs search using Qdrant vector database
        """
        with observability.measure_agent_performance("vec", state):
            result = self.vector_rag.retrieve_documents(state)
            return result
    
    def graph_rag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Graph RAG agent node - Simple graph queries
        Performs knowledge graph queries using Neo4j
        """
        with observability.measure_agent_performance("graph", state):
            result = self.graph_rag.extract_and_query(state)
            return result
    
    def validator_node(self, state: WorkflowState) -> WorkflowState:
        """
        Validator agent node - Simple consistency checking
        Happy path: validation should generally pass
        """
        with observability.measure_agent_performance("val", state):
            result = self.validator.validate_results(state)
            return result
    
    def synthesizer_node(self, state: WorkflowState) -> WorkflowState:
        """
        Answer synthesis agent node - Final answer composition
        Creates comprehensive answer without citations
        """
        with observability.measure_agent_performance("ans", state):
            result = self.synthesizer.synthesize_answer(state)
            return result
    
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
            # This should never happen with function calling OrchestratorAgent
            # but provide fallback for safety
            return "both_vector_first"
    
    def check_if_graph_needed(self, state: WorkflowState) -> str:
        """
        Simple check if we need to continue to graph after vector retrieval
        """
        route = state.get("route", "")
        if route == "both":
            return "continue_to_graph"
        else:
            return "continue_to_validator"
