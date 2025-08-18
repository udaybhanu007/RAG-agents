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

from enhanced_query_analyzer import EnhancedQueryAnalyzer
from dynamic_tool_selector import DynamicToolSelector
from execution_planner import ExecutionPlanner
from simple_agentic_agents import AgenticOrchestratorAgent, LearningMemory
from base_classes import WorkflowState, QueryResult, AgentResult
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
        self.learning_memory = None
        self.initialized = False
        logger.info("enhanced_agentic_app_created")
    
    def initialize(self, llm, vector_store=None, graph_store=None):
        """Initialize the enhanced agentic system"""
        logger.info("initializing_enhanced_agentic_system")
        try:
            # Initialize all enhanced components
            self.query_analyzer = EnhancedQueryAnalyzer(llm)
            self.tool_selector = DynamicToolSelector(llm)
            self.execution_planner = ExecutionPlanner(llm)
            self.learning_memory = LearningMemory()
            self.orchestrator_agent = AgenticOrchestratorAgent(
                llm=llm,
                vector_store=vector_store,
                graph_store=graph_store,
                learning_memory=self.learning_memory
            )
            
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
            tool_selection = self.tool_selector.select_tools_with_reasoning(query, analysis)
            
            # Step 3: Execution planning
            logger.info("step_3_execution_planning")
            execution_plan = self.execution_planner.create_comprehensive_plan(query, analysis, tool_selection)
            
            # Step 4: Execute through orchestrator agent
            logger.info("step_4_orchestrator_execution")
            state = WorkflowState(
                user_query=query,
                sanitized_query=query,
                query_result=QueryResult(final_answer="", sources=[], confidence_score=0.0),
                agent_results=[],
                reasoning_steps=[],
                current_step="comprehensive_processing",
                is_complete=False,
                metadata={
                    "analysis": analysis,
                    "tool_selection": tool_selection, 
                    "execution_plan": execution_plan
                }
            )
            
            # Process through agentic orchestrator
            result = self.orchestrator_agent.reason_and_plan(state)
            
            logger.info("enhanced_query_processing_completed", 
                       has_answer=bool(result.get("final_answer")),
                       has_error=result.get("error", False),
                       confidence=result.get("confidence_score", 0.0),
                       autonomous_reasoning=True)
            return result
            
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
            return self.enhanced_system.get_agentic_capabilities_status()
        except Exception as e:
            logger.error("get_system_status_failed", error=str(e))
            return {"status": "Error", "message": str(e)}
    
    def reset_learning(self):
        """Reset all learning data for fresh start"""
        if self.initialized:
            try:
                self.enhanced_system.reset_learning_state()
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
            status = self.enhanced_system.get_agentic_capabilities_status()
            learning_insights = status.get("learning_insights", {})
            component_stats = status.get("component_capabilities", {})
            
            return {
                "learning_summary": {
                    "total_adaptations": learning_insights.get("recent_adaptations", 0),
                    "processing_patterns": learning_insights.get("processing_patterns", {}),
                    "performance_trends": learning_insights.get("performance_trends", {})
                },
                "component_learning": {
                    "query_analyzer": {
                        "total_analyses": component_stats.get("query_analyzer", {}).get("total_analyses", 0),
                        "medical_query_percentage": component_stats.get("query_analyzer", {}).get("medical_query_percentage", 0),
                        "complexity_patterns": component_stats.get("query_analyzer", {}).get("complexity_distribution", {})
                    },
                    "tool_selector": {
                        "total_selections": component_stats.get("tool_selector", {}).get("total_selections", 0),
                        "tool_usage_patterns": component_stats.get("tool_selector", {}).get("tool_usage_distribution", {}),
                        "most_effective_tool": component_stats.get("tool_selector", {}).get("most_used_tool", "none")
                    },
                    "execution_planner": {
                        "total_executions": component_stats.get("execution_planner", {}).get("total_executions", 0),
                        "success_rate": component_stats.get("execution_planner", {}).get("average_success_rate", 0),
                        "most_effective_tools": component_stats.get("execution_planner", {}).get("most_effective_tools", {})
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
