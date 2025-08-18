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
from updated_agents.simple_agentic_agents import AgenticOrchestratorAgent, LearningMemory, SimpleAgenticWorkflow
from updated_agents.base_classes import WorkflowState, QueryResult, AgentResult
from core.logging_config import get_logger

# Initialize logger for the application
logger = get_logger("simple_agentic_app")

class EnhancedAgenticRAGApplication:
    """
    Main Application Class for Enhanced Agentic RAG System
    
    This class provides the primary interface for the truly agentic RAG system
    with comprehensive capabilities:
    
    - Deep LLM-based query analysis (medical relevance, complexity, sub-questions)
    - Dynamic reasoning-based tool selection
    - Adaptive execution planning with contingencies
    - Learning from execution outcomes
    - Maximum security integration with core modules
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
            
            # Initialize the complete workflow system instead of just orchestrator
            self.workflow = SimpleAgenticWorkflow(llm, vector_client, graph_store)
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
            
            # Initialize the complete workflow system instead of just orchestrator
            self.workflow = SimpleAgenticWorkflow(llm, vector_store, graph_store)
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
        logger.info("enhanced_query_processing_started", query_length=len(query))
        
        if not self.initialized:
            logger.warning("query_processing_failed_not_initialized")
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
            logger.info("step_1_comprehensive_query_analysis")
            analysis = self.query_analyzer.analyze_query_comprehensive(query)
            
            # Step 2: Dynamic tool selection
            logger.info("step_2_dynamic_tool_selection")
            tool_selection = self.tool_selector.select_tools_with_reasoning(analysis)
            
            # Step 3: Create base execution plan
            logger.info("step_3_base_execution_plan")
            base_execution_plan = self.tool_selector.create_execution_plan(analysis, tool_selection)
            
            # Step 4: Create comprehensive execution plan
            logger.info("step_4_comprehensive_execution_planning")
            execution_plan = self.execution_planner.create_comprehensive_plan(analysis, tool_selection, base_execution_plan)
            
            # Step 5: Execute through orchestrator agent  
            logger.info("step_4_orchestrator_execution")
            state = WorkflowState(
                query=query,
                sanitized_query=query,
                final_answer="",
                sources=[],
                confidence_score=0.0,
                agent_results=[],
                reasoning_steps=[],
                current_step="comprehensive_processing",
                is_complete=False,
                analysis=analysis,
                tool_selection=tool_selection,
                execution_plan=execution_plan
            )
            
            # Process through agentic workflow for complete pipeline
            result = self.workflow.process_query(query)
            
            logger.info("enhanced_query_processing_completed", 
                       has_answer=bool(result.get("answer")),
                       has_error=result.get("error", False),
                       confidence=result.get("confidence_score", 0.0),
                       autonomous_reasoning=True)
            
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
            logger.error("enhanced_query_processing_failed", error=str(e))
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
    
    def analyze_query_deep(self, query: str) -> Dict[str, Any]:
        """
        Perform deep analysis of query without full processing
        
        This provides insights into:
        - Medical relevance and domain classification
        - Query complexity and information type
        - Required tools and processing steps
        - Sub-questions and relationships
        """
        logger.info("deep_query_analysis_requested", query_length=len(query))
        
        if not self.initialized:
            return {"error": "System not initialized"}
        
        try:
            # Use the enhanced query analyzer for deep analysis
            analysis = self.query_analyzer.analyze_query_comprehensive(query)
            
            # Convert to dictionary format for return
            analysis_result = {
                "medical_domain": {
                    "is_medical": analysis.medical_domain.is_medical,
                    "domain": analysis.medical_domain.medical_domain,
                    "confidence": analysis.medical_domain.confidence_score,
                    "entities": analysis.medical_domain.medical_entities,
                    "clinical_context": analysis.medical_domain.clinical_context
                },
                "complexity": {
                    "level": analysis.complexity.complexity_level,
                    "reasoning": analysis.complexity.reasoning,
                    "processing_time": analysis.complexity.estimated_processing_time,
                    "multiple_steps": analysis.complexity.requires_multiple_steps
                },
                "information_seeking": {
                    "type": analysis.information_seeking.information_type,
                    "needs": analysis.information_seeking.specific_needs,
                    "requires_relationships": analysis.information_seeking.requires_relationships,
                    "requires_quantitative": analysis.information_seeking.requires_quantitative_data,
                    "temporal": analysis.information_seeking.temporal_aspect
                },
                "sub_questions": {
                    "has_multiple": analysis.sub_question_analysis.has_multiple_questions,
                    "questions": analysis.sub_question_analysis.sub_questions,
                    "dependencies": analysis.sub_question_analysis.question_dependencies,
                    "order": analysis.sub_question_analysis.processing_order
                },
                "tool_requirements": {
                    "recommended": analysis.tool_requirements.recommended_tools,
                    "priorities": analysis.tool_requirements.tool_priorities,
                    "reasoning": analysis.tool_requirements.tool_reasoning,
                    "fallbacks": analysis.tool_requirements.fallback_options
                },
                "overall_strategy": analysis.overall_strategy,
                "confidence": analysis.overall_confidence
            }
            
            logger.info("deep_query_analysis_completed",
                       is_medical=analysis.medical_domain.is_medical,
                       complexity=analysis.complexity.complexity_level)
            return analysis_result
            
        except Exception as e:
            logger.error("deep_query_analysis_failed", error=str(e))
            return {"error": f"Analysis failed: {str(e)}"}
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights from the system"""
        if not self.initialized:
            return {"error": "System not initialized"}
        
        try:
            # Get insights from learning memory
            learning_insights = self.learning_memory.get_insights() if hasattr(self.learning_memory, 'get_insights') else {}
            
            # Get component statistics
            component_stats = {}
            if hasattr(self.query_analyzer, 'get_analysis_statistics'):
                component_stats["query_analyzer"] = self.query_analyzer.get_analysis_statistics()
            if hasattr(self.tool_selector, 'get_selection_statistics'):
                component_stats["tool_selector"] = self.tool_selector.get_selection_statistics()
            if hasattr(self.execution_planner, 'get_execution_statistics'):
                component_stats["execution_planner"] = self.execution_planner.get_execution_statistics()
            
            return {
                "system_learning": {
                    "total_adaptations": learning_insights.get("recent_adaptations", 0),
                    "processing_patterns": learning_insights.get("processing_patterns", {}),
                    "performance_trends": learning_insights.get("performance_trends", {})
                },
                "component_learning": component_stats,
                "intelligence_indicators": {
                    "autonomous_decision_making": learning_insights.get("recent_adaptations", 0) > 0,
                    "pattern_recognition": bool(learning_insights.get("processing_patterns", {})),
                    "adaptive_behavior": bool(learning_insights.get("performance_trends", {})),
                    "continuous_learning": "Learning from each interaction"
                }
            }
            
        except Exception as e:
            logger.error("get_learning_insights_failed", error=str(e))
            return {"error": f"Failed to get learning insights: {str(e)}"}
    
    def reset_learning(self):
        """Reset learning memory"""
        if not self.initialized:
            return {"error": "System not initialized"}
        
        try:
            if hasattr(self.learning_memory, 'reset'):
                self.learning_memory.reset()
            logger.info("learning_memory_reset")
            return {"status": "success", "message": "Learning memory reset"}
        except Exception as e:
            logger.error("reset_learning_failed", error=str(e))
            return {"error": f"Failed to reset learning: {str(e)}"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all agentic capabilities"""
        if not self.initialized:
            return {
                "status": "Not Initialized",
                "agentic_capabilities": {
                    "comprehensive_analysis": False,
                    "dynamic_tool_selection": False,
                    "adaptive_execution": False,
                    "learning_enabled": False,
                    "contingency_handling": False,
                    "security_integration": False
                }
            }
        
        try:
            return {
                "status": "Active",
                "agentic_capabilities": {
                    "comprehensive_analysis": self.query_analyzer is not None,
                    "dynamic_tool_selection": self.tool_selector is not None,
                    "adaptive_execution": self.execution_planner is not None,
                    "learning_enabled": self.learning_memory is not None,
                    "contingency_handling": True,
                    "security_integration": True
                },
                "components": {
                    "query_analyzer": self.query_analyzer is not None,
                    "tool_selector": self.tool_selector is not None,
                    "execution_planner": self.execution_planner is not None,
                    "orchestrator_agent": self.orchestrator_agent is not None,
                    "learning_memory": self.learning_memory is not None
                }
            }
        except Exception as e:
            logger.error("get_system_status_failed", error=str(e))
            return {"status": "Error", "message": str(e)}
    
    def reset_learning(self):
        """Reset all learning data for fresh start"""
        if self.initialized:
            try:
                if self.learning_memory:
                    self.learning_memory.clear_all_learning()
                logger.info("learning_state_reset_successful")
                return {"status": "success", "message": "All learning state reset successfully"}
            except Exception as e:
                logger.error("learning_state_reset_failed", error=str(e))
                return {"status": "error", "message": f"Reset failed: {str(e)}"}
        return {"status": "error", "message": "System not initialized"}
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights into what the system has learned"""
        if not self.initialized:
            return {"learning_data": "System not initialized"}
        
        try:
            learning_insights = {}
            if self.learning_memory:
                learning_insights = {
                    "recent_adaptations": self.learning_memory.adaptation_count,
                    "processing_patterns": self.learning_memory.query_patterns,
                    "performance_trends": self.learning_memory.routing_performance
                }
            
            return {
                "learning_summary": {
                    "total_adaptations": learning_insights.get("recent_adaptations", 0),
                    "processing_patterns": learning_insights.get("processing_patterns", {}),
                    "performance_trends": learning_insights.get("performance_trends", {})
                },
                "component_learning": {
                    "query_analyzer": {
                        "total_analyses": 0,
                        "medical_query_percentage": 0,
                        "complexity_patterns": {}
                    },
                    "tool_selector": {
                        "total_selections": 0,
                        "tool_usage_patterns": {},
                        "most_effective_tool": "none"
                    },
                    "execution_planner": {
                        "total_executions": 0,
                        "success_rate": 0,
                        "most_effective_tools": {}
                    }
                },
                "intelligence_indicators": {
                    "autonomous_decision_making": learning_insights.get("recent_adaptations", 0) > 0,
                    "pattern_recognition": bool(learning_insights.get("processing_patterns", {})),
                    "adaptive_behavior": bool(learning_insights.get("performance_trends", {})),
                    "continuous_learning": "Learning from each interaction"
                }
            }
            
        except Exception as e:
            logger.error("get_learning_insights_failed", error=str(e))
            return {"error": f"Failed to get learning insights: {str(e)}"}
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get detailed information about processing capabilities"""
        return {
            "query_analysis_capabilities": {
                "medical_domain_detection": "Advanced LLM-based classification",
                "complexity_assessment": "Multi-factor complexity analysis",
                "information_type_classification": "Factual, Comparative, Analytical, Procedural, Diagnostic",
                "sub_question_decomposition": "Automatic breakdown of complex queries",
                "temporal_analysis": "Current, Historical, Trending, Predictive"
            },
            "tool_selection_capabilities": {
                "dynamic_reasoning": "LLM-assisted tool selection reasoning",
                "performance_learning": "Tool effectiveness tracking and adaptation",
                "contingency_planning": "Multiple fallback strategies",
                "resource_optimization": "Efficient resource allocation"
            },
            "execution_capabilities": {
                "adaptive_planning": "Dynamic execution plan creation",
                "contingency_handling": "Built-in error recovery mechanisms",
                "quality_monitoring": "Real-time quality assessment",
                "parallel_execution": "Concurrent tool execution where possible"
            },
            "learning_capabilities": {
                "pattern_recognition": "Query and response pattern learning",
                "performance_adaptation": "Tool selection improvement over time",
                "strategy_optimization": "Execution strategy refinement",
                "quality_improvement": "Continuous quality enhancement"
            },
            "security_features": {
                "input_sanitization": "Comprehensive input cleaning and validation",
                "prompt_injection_detection": "Advanced injection attempt detection",
                "secure_llm_interactions": "Protected LLM communication",
                "error_handling": "Graceful error recovery and reporting"
            }
        }

# Global application instance
enhanced_agentic_app = EnhancedAgenticRAGApplication()
