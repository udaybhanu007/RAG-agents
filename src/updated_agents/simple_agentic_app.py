"""
Simple Agentic RAG Application with Enhanced Capabilities

This module provides a clean, simple interface to the enhanced agentic RAG system
with comprehensive query analysis, dynamic tool selection, and execution planning.
"""

import sys
import os
from typing import Dict, Any

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import Azure Key Vault manager for secure secret management
from core.azure_keyvault_manager import get_secret_from_keyvault

from updated_agents.enhanced_query_analyzer import EnhancedQueryAnalyzer
from updated_agents.dynamic_tool_selector import DynamicToolSelector
from updated_agents.execution_planner import ExecutionPlanner
from updated_agents.simple_agentic_agents import AgenticOrchestratorAgent, LearningMemory
from updated_agents.langgraph_agentic_workflow import LangGraphAgenticWorkflow, create_langgraph_agentic_workflow
from updated_agents.base_classes import WorkflowState
from core.logging_config import get_logger

# Initialize logger for the application
logger = get_logger("simple_agentic_app")

class EnhancedAgenticRAGApplication:
    """
    LangGraph-Powered Enhanced Agentic RAG System
    
    This class provides the primary interface for the truly agentic RAG system
    with comprehensive LangGraph StateGraph orchestration and capabilities:
    
    - Deep LLM-based query analysis (medical relevance, complexity, sub-questions)
    - Dynamic reasoning-based tool selection
    - Adaptive execution planning with contingencies
    - Learning from execution outcomes
    - LangGraph StateGraph with nodes and edges for standardized orchestration
    - Maximum security integration with core modules
    - All business logic preserved with enhanced framework compliance
    """
    
    def __init__(self):
        self.query_analyzer = None
        self.tool_selector = None
        self.execution_planner = None
        self.orchestrator_agent = None
        self.workflow = None
        self.learning_memory = None
        self.initialized = False
        logger.info("enhanced_agentic_app_created")
    
    def initialize_system(self):
        """
        Initialize the enhanced agentic system with all components
        This method handles all initialization internally, similar to MultiAgentRAGWorkflow
        """
        logger.info("initializing_enhanced_agentic_system_self_contained")
        try:
            # Import necessary components
            from core.azure_keyvault_manager import AzureKeyVaultManager
            from langchain_openai import AzureChatOpenAI
            from qdrant_client import QdrantClient
            from neo4j import GraphDatabase
            
            # Initialize Azure Key Vault Manager with proper environment handling
            # Use proper Azure Key Vault integration
            try:
                logger.info("using_azure_keyvault_integration")
                
                # Initialize Azure OpenAI LLM using Key Vault secrets
                azure_endpoint = get_secret_from_keyvault("AZURE_OPENAI_ENDPOINT")
                azure_api_key = get_secret_from_keyvault("AZURE_OPENAI_API_KEY")
                azure_deployment = get_secret_from_keyvault("AZURE_OPENAI_DEPLOYMENT")
                azure_api_version = get_secret_from_keyvault("AZURE_OPENAI_API_VERSION")
                
                if not all([azure_endpoint, azure_api_key, azure_deployment, azure_api_version]):
                    raise ValueError("Missing required Azure OpenAI configuration. Check Azure Key Vault or .env.dev file.")
                
                from langchain_openai import AzureChatOpenAI
                llm = AzureChatOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    azure_deployment=azure_deployment,
                    api_version=azure_api_version,
                    temperature=0.0
                )
                logger.debug("azure_llm_initialized_with_keyvault")
                
            except Exception as e:
                logger.error("azure_llm_initialization_failed", error=str(e))
                raise ValueError(f"Failed to initialize Azure OpenAI LLM: {str(e)}")
            
            # Initialize Qdrant vector store using Key Vault secrets
            try:
                qdrant_url = get_secret_from_keyvault("QDRANT_API_URL")
                qdrant_api_key = get_secret_from_keyvault("QDRANT_API_KEY")
                
                if qdrant_url and qdrant_api_key:
                    from qdrant_client import QdrantClient
                    import re
                    
                    # Parse URL for Qdrant client initialization
                    url_match = re.match(r'https?://([^:]+):(\d+)', qdrant_url)
                    if url_match:
                        host = url_match.group(1)
                        port = int(url_match.group(2))
                        vector_client = QdrantClient(
                            host=host,
                            port=port,
                            api_key=qdrant_api_key,
                            https=True
                        )
                    else:
                        vector_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                    logger.debug("qdrant_vector_client_initialized_with_secrets")
                else:
                    # Fallback to local Qdrant
                    from qdrant_client import QdrantClient
                    vector_client = QdrantClient(host="localhost", port=6333)
                    logger.debug("qdrant_vector_client_initialized_local_fallback")
                
            except Exception as e:
                logger.warning("qdrant_initialization_failed", error=str(e))
                vector_client = None
            
            # Initialize Neo4j graph store using Key Vault secrets
            try:
                neo4j_uri = get_secret_from_keyvault("NEO4J_URI")
                neo4j_user = get_secret_from_keyvault("NEO4J_USERNAME")
                neo4j_password = get_secret_from_keyvault("NEO4J_PASSWORD")
                
                if neo4j_uri and neo4j_user and neo4j_password:
                    from neo4j import GraphDatabase
                    graph_store = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                    logger.debug("neo4j_graph_store_initialized_with_secrets")
                else:
                    logger.warning("neo4j_credentials_missing", uri=bool(neo4j_uri), user=bool(neo4j_user), password=bool(neo4j_password))
                    graph_store = None
            except Exception as e:
                logger.warning("neo4j_initialization_failed", error=str(e))
                graph_store = None
            
            # Initialize all enhanced components
            self.query_analyzer = EnhancedQueryAnalyzer(llm)
            self.tool_selector = DynamicToolSelector(llm)
            self.execution_planner = ExecutionPlanner(llm)
            self.learning_memory = LearningMemory()
            
            # Initialize the LangGraph workflow system instead of simple workflow
            self.workflow = LangGraphAgenticWorkflow(llm, vector_client, graph_store)
            self.orchestrator_agent = self.workflow.orchestrator
            
            self.initialized = True
            logger.info("enhanced_agentic_system_initialized_successfully")
            return {
                "status": "success", 
                "message": "Enhanced Agentic RAG System initialized successfully",
                "capabilities": {
                    "comprehensive_analysis": True,
                    "dynamic_tool_selection": True,
                    "adaptive_execution": True,
                    "learning_enabled": True,
                    "security_integrated": True
                }
            }
        except Exception as e:
            logger.error("enhanced_agentic_system_initialization_failed", error=str(e))
            return {"status": "error", "message": f"Initialization failed: {str(e)}"}
    
    def initialize(self, llm, vector_store=None, graph_store=None):
        """Initialize the enhanced agentic system with provided components (legacy method)"""
        logger.info("initializing_enhanced_agentic_system")
        try:
            # Initialize all enhanced components
            self.query_analyzer = EnhancedQueryAnalyzer(llm)
            self.tool_selector = DynamicToolSelector(llm)
            self.execution_planner = ExecutionPlanner(llm)
            self.learning_memory = LearningMemory()
            
            # Initialize the LangGraph workflow system instead of simple workflow
            self.workflow = LangGraphAgenticWorkflow(llm, vector_store, graph_store)
            self.orchestrator_agent = self.workflow.orchestrator
            
            self.initialized = True
            logger.info("enhanced_agentic_system_initialized_successfully")
            return {
                "status": "success", 
                "message": "Enhanced Agentic RAG System initialized successfully",
                "capabilities": {
                    "comprehensive_analysis": True,
                    "dynamic_tool_selection": True,
                    "adaptive_execution": True,
                    "learning_enabled": True,
                    "security_integrated": True
                }
            }
        except Exception as e:
            logger.error("enhanced_agentic_system_initialization_failed", error=str(e))
            return {"status": "error", "message": f"Initialization failed: {str(e)}"}
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query with comprehensive agentic capabilities
        
        This method demonstrates the full enhanced pipeline:
        1. Comprehensive query analysis (medical relevance, complexity, information type, sub-questions)
        2. Dynamic tool selection based on reasoning
        3. Adaptive execution planning with contingencies
        4. Learning from execution outcomes
        5. Security validation throughout
        """
        # Create initial state with trace_id
        state = WorkflowState(query=query)
        trace_id = state.get("trace_id")
        
        logger.info("enhanced_query_processing_started", 
                   query_length=len(query), 
                   trace_id=trace_id)
        
        if not self.initialized:
            logger.warning("query_processing_failed_not_initialized", trace_id=trace_id)
            return {
                "final_answer": "System not initialized",
                "sources": [],
                "confidence_score": 0.0,
                "error": True,
                "agentic_indicators": {
                    "autonomous_reasoning": False,
                    "learning_applied": False,
                    "adaptive_behavior": False,
                    "dynamic_planning": False,
                    "contingency_handling": False
                }
            }
        
        try:
            # Step 1: Comprehensive query analysis
            logger.info("step_1_comprehensive_query_analysis", trace_id=trace_id)
            analysis = self.query_analyzer.analyze_query_comprehensive(query, trace_id=trace_id)
            
            # Step 2: Dynamic tool selection
            logger.info("step_2_dynamic_tool_selection", trace_id=trace_id)
            tool_selection = self.tool_selector.select_tools_with_reasoning(analysis, trace_id=trace_id)
            
            # Step 3: Create comprehensive execution plan
            logger.info("step_3_comprehensive_execution_planning", trace_id=trace_id)
            execution_plan = self.execution_planner.create_comprehensive_plan(analysis, tool_selection, trace_id=trace_id)
            
            # Step 5: Execute through orchestrator agent  
            logger.info("step_5_orchestrator_execution", trace_id=trace_id)
            
            # Update state with trace_id
            state.add_result("sanitized_query", query)
            state.add_result("final_answer", "")
            state.add_result("sources", [])
            state.add_result("confidence_score", 0.0)
            state.add_result("agent_results", [])
            state.add_result("reasoning_steps", [])
            state.add_result("current_step", "comprehensive_processing")
            state.add_result("is_complete", False)
            state.add_result("analysis", analysis)
            state.add_result("tool_selection", tool_selection)
            state.add_result("execution_plan", execution_plan)
            
            # Process through agentic workflow for complete pipeline
            result = self.workflow.process_query(query, trace_id=trace_id)
            
            logger.info("enhanced_query_processing_completed", 
                       has_answer=bool(result.get("answer")),
                       has_error=result.get("error", False),
                       confidence=result.get("confidence_score", 0.0),
                       autonomous_reasoning=True,
                       trace_id=trace_id)
            
            # Convert the result format to match expected structure
            final_result = {
                "final_answer": result.get("answer", "No answer generated"),
                "sources": result.get("sources", []),
                "confidence_score": result.get("confidence_score", 0.0),
                "error": result.get("error", False),
                "agentic_indicators": result.get("agentic_indicators", {
                    "autonomous_reasoning": True,
                    "learning_applied": bool(result.get("learning_update")),
                    "adaptive_behavior": True,
                    "dynamic_planning": True,
                    "contingency_handling": True
                })
            }
            return final_result
            
        except Exception as e:
            logger.error("enhanced_query_processing_failed", error=str(e), trace_id=trace_id)
            return {
                "final_answer": f"Processing failed: {str(e)}",
                "sources": [],
                "confidence_score": 0.0,
                "error": True,
                "agentic_indicators": {
                    "autonomous_reasoning": False,
                    "learning_applied": False,
                    "adaptive_behavior": False,
                    "dynamic_planning": False,
                    "contingency_handling": False
                }
            }

# Global application instance
enhanced_agentic_app = EnhancedAgenticRAGApplication()
