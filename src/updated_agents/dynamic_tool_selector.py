"""
Dynamic Tool Selector with Reasoning-based Selection

This module provides intelligent tool selection based on comprehensive query analysis.
It uses reasoning to select the most appropriate tools dynamically rather than using
static rules.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic.v1 import BaseModel, Field
from abc import ABC, abstractmethod

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from updated_agents.enhanced_query_analyzer import ComprehensiveQueryAnalysis, ToolRequirementAnalysis
from core.input_sanitization import secure_llm_interaction
from core.logging_config import get_logger
from core.observability import traceable, get_traceable_config

logger = get_logger("dynamic_tool_selector")

class ToolCapability(BaseModel):
    """Definition of a tool's capabilities"""
    tool_name: str = Field(description="Name of the tool")
    primary_purpose: str = Field(description="Primary purpose of the tool")
    strengths: List[str] = Field(description="What this tool excels at")
    limitations: List[str] = Field(description="Tool limitations")
    optimal_conditions: List[str] = Field(description="Conditions where tool performs best")
    resource_requirements: str = Field(description="Resource requirements (Low, Medium, High)")
    typical_response_time: str = Field(description="Expected response time category")

class ToolSelectionReasoning(BaseModel):
    """Reasoning behind tool selection"""
    selected_tools: List[str] = Field(description="Tools selected for execution")
    tool_order: List[str] = Field(description="Order of tool execution")
    reasoning_steps: List[str] = Field(description="Step-by-step reasoning process")
    confidence_scores: Dict[str, float] = Field(description="Confidence in each tool selection")
    contingency_plans: List[str] = Field(description="Backup plans if primary tools fail")
    resource_allocation: Dict[str, str] = Field(description="Resource allocation strategy")
    expected_outcomes: Dict[str, str] = Field(description="Expected outcomes from each tool")

class ToolExecutionPlan(BaseModel):
    """Detailed execution plan for selected tools"""
    plan_id: str = Field(description="Unique plan identifier")
    primary_strategy: str = Field(description="Primary execution strategy")
    tool_sequence: List[Dict[str, Any]] = Field(description="Ordered sequence of tool executions")
    parallel_opportunities: List[List[str]] = Field(description="Tools that can run in parallel")
    checkpoints: List[str] = Field(description="Validation checkpoints during execution")
    fallback_strategies: List[Dict[str, Any]] = Field(description="Fallback strategies")
    success_criteria: List[str] = Field(description="Criteria for measuring success")
    estimated_duration: str = Field(description="Estimated total execution time")

class AbstractTool(ABC):
    """Abstract base class for all tools"""
    
    def __init__(self, name: str):
        self.name = name
        self.execution_count = 0
        self.success_rate = 1.0
        self.average_duration = 0.0
    
    @abstractmethod
    def get_capabilities(self) -> ToolCapability:
        """Get tool capabilities"""
        pass
    
    @abstractmethod
    def can_handle_query(self, analysis: ComprehensiveQueryAnalysis) -> float:
        """
        Assess if this tool can handle the query
        Returns confidence score 0.0-1.0
        """
        pass
    
    def update_performance_metrics(self, success: bool, duration: float):
        """Update tool performance metrics"""
        self.execution_count += 1
        if success:
            total_successes = self.success_rate * (self.execution_count - 1) + 1
            self.success_rate = total_successes / self.execution_count
        else:
            total_successes = self.success_rate * (self.execution_count - 1)
            self.success_rate = total_successes / self.execution_count
        
        # Update average duration
        total_duration = self.average_duration * (self.execution_count - 1) + duration
        self.average_duration = total_duration / self.execution_count

class VectorSearchTool(AbstractTool):
    """Vector search tool implementation"""
    
    def __init__(self):
        super().__init__("vector_search")
    
    def get_capabilities(self) -> ToolCapability:
        return ToolCapability(
            tool_name="vector_search",
            primary_purpose="Semantic similarity search in document collections",
            strengths=[
                "Fast semantic similarity matching",
                "Good for factual information retrieval",
                "Handles large document collections efficiently",
                "Excellent for finding relevant context"
            ],
            limitations=[
                "Limited relationship understanding",
                "May miss indirect connections",
                "Not optimal for complex reasoning",
                "Struggles with multi-hop queries"
            ],
            optimal_conditions=[
                "Single-concept queries",
                "Factual information requests",
                "Large document collections",
                "Semantic similarity is key"
            ],
            resource_requirements="Medium",
            typical_response_time="Fast"
        )
    
    def can_handle_query(self, analysis: ComprehensiveQueryAnalysis) -> float:
        """Assess capability to handle query"""
        confidence = 0.5  # Base confidence
        
        # Boost for factual queries
        if analysis.information_seeking.information_type == "Factual":
            confidence += 0.3
        
        # Boost for medical queries (good document coverage)
        if analysis.medical_domain.is_medical:
            confidence += 0.2
        
        # Reduce for complex relationship queries
        if analysis.information_seeking.requires_relationships:
            confidence -= 0.2
        
        # Reduce for multi-faceted complexity
        if analysis.complexity.complexity_level == "Multi-faceted":
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))

class GraphSearchTool(AbstractTool):
    """Graph search tool implementation"""
    
    def __init__(self):
        super().__init__("graph_search")
    
    def get_capabilities(self) -> ToolCapability:
        return ToolCapability(
            tool_name="graph_search",
            primary_purpose="Relationship and connection analysis in structured data",
            strengths=[
                "Excellent relationship discovery",
                "Multi-hop connection analysis",
                "Pattern recognition in connections",
                "Complex query reasoning"
            ],
            limitations=[
                "Requires structured data",
                "Slower for simple factual queries",
                "Limited by data modeling quality",
                "Higher computational overhead"
            ],
            optimal_conditions=[
                "Relationship-focused queries",
                "Multi-entity analysis",
                "Pattern discovery needs",
                "Complex reasoning requirements"
            ],
            resource_requirements="High",
            typical_response_time="Standard"
        )
    
    def can_handle_query(self, analysis: ComprehensiveQueryAnalysis) -> float:
        """Assess capability to handle query"""
        confidence = 0.4  # Base confidence
        
        # High boost for relationship queries
        if analysis.information_seeking.requires_relationships:
            confidence += 0.4
        
        # Boost for analytical and comparative queries
        if analysis.information_seeking.information_type in ["Analytical", "Comparative"]:
            confidence += 0.3
        
        # Boost for complex queries
        if analysis.complexity.complexity_level in ["Complex", "Multi-faceted"]:
            confidence += 0.2
        
        # Boost for multi-question scenarios
        if analysis.sub_questions.has_multiple_questions:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))

class HybridSearchTool(AbstractTool):
    """Hybrid search tool (both vector and graph)"""
    
    def __init__(self):
        super().__init__("hybrid_search")
    
    def get_capabilities(self) -> ToolCapability:
        return ToolCapability(
            tool_name="hybrid_search",
            primary_purpose="Combined semantic and relationship analysis",
            strengths=[
                "Comprehensive coverage",
                "Best of both approaches",
                "Handles diverse query types",
                "High accuracy potential"
            ],
            limitations=[
                "Higher resource consumption",
                "Longer execution time",
                "Complex result synthesis",
                "Potential information redundancy"
            ],
            optimal_conditions=[
                "Complex multi-faceted queries",
                "High accuracy requirements",
                "Uncertain query type",
                "Comprehensive analysis needed"
            ],
            resource_requirements="High",
            typical_response_time="Extended"
        )
    
    def can_handle_query(self, analysis: ComprehensiveQueryAnalysis) -> float:
        """Assess capability to handle query"""
        confidence = 0.7  # High base confidence for comprehensive approach
        
        # Boost for complex queries
        if analysis.complexity.complexity_level in ["Complex", "Multi-faceted"]:
            confidence += 0.2
        
        # Boost for analytical queries
        if analysis.information_seeking.information_type == "Analytical":
            confidence += 0.1
        
        # Boost for multi-question scenarios
        if analysis.sub_questions.has_multiple_questions:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))

class DynamicToolSelector:
    """
    Dynamic Tool Selector with Reasoning-based Selection
    
    This class intelligently selects tools based on comprehensive query analysis
    and reasoning about tool capabilities versus query requirements.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.available_tools = {
            "vector_search": VectorSearchTool(),
            "graph_search": GraphSearchTool(),
            "hybrid_search": HybridSearchTool()
        }
        self.selection_history = []
        self.performance_tracking = {}
        logger.info("dynamic_tool_selector_initialized", 
                   available_tools=list(self.available_tools.keys()))
    
    @traceable(**get_traceable_config("DynamicToolSelector"))
    def select_tools_with_reasoning(self, analysis: ComprehensiveQueryAnalysis) -> ToolSelectionReasoning:
        """
        Select tools using comprehensive reasoning about query requirements
        and tool capabilities
        
        Args:
            analysis: Comprehensive query analysis
            
        Returns:
            ToolSelectionReasoning: Detailed tool selection with reasoning
        """
        logger.info("tool_selection_reasoning_started", 
                   analysis_id=analysis.query_id,
                   query_complexity=analysis.complexity.complexity_level)
        
        try:
            # Step 1: Assess each tool's capability for this query
            tool_assessments = self._assess_tool_capabilities(analysis)
            
            # Step 2: Apply learned performance data
            adjusted_assessments = self._apply_performance_learning(tool_assessments, analysis)
            
            # Step 3: Use LLM reasoning for final selection
            llm_reasoning = self._llm_assisted_selection(analysis, adjusted_assessments)
            
            # Step 4: Create execution strategy
            selection_reasoning = self._create_selection_reasoning(
                analysis, adjusted_assessments, llm_reasoning
            )
            
            # Step 5: Record selection for learning
            self._record_selection_decision(analysis, selection_reasoning)
            
            logger.info("tool_selection_reasoning_completed",
                       selected_tools=selection_reasoning.selected_tools,
                       primary_strategy=selection_reasoning.tool_order[0] if selection_reasoning.tool_order else "none")
            
            return selection_reasoning
            
        except Exception as e:
            logger.error("tool_selection_reasoning_failed", error=str(e))
            # Return fallback selection
            return self._create_fallback_selection(analysis)
    
    @traceable(**get_traceable_config("DynamicToolSelector"))
    def create_execution_plan(self, analysis: ComprehensiveQueryAnalysis, 
                            selection: ToolSelectionReasoning) -> ToolExecutionPlan:
        """
        Create detailed execution plan based on tool selection
        
        Args:
            analysis: Query analysis
            selection: Tool selection reasoning
            
        Returns:
            ToolExecutionPlan: Detailed execution plan
        """
        logger.info("execution_plan_creation_started", 
                   analysis_id=analysis.query_id,
                   selected_tools=selection.selected_tools)
        
        plan_id = f"plan_{analysis.query_id}_{datetime.now().strftime('%H%M%S')}"
        
        # Determine execution strategy
        primary_strategy = self._determine_execution_strategy(analysis, selection)
        
        # Create tool sequence
        tool_sequence = self._create_tool_sequence(selection)
        
        # Identify parallel opportunities
        parallel_opportunities = self._identify_parallel_opportunities(selection)
        
        # Define checkpoints
        checkpoints = self._define_execution_checkpoints(analysis, selection)
        
        # Create fallback strategies
        fallback_strategies = self._create_fallback_strategies(analysis, selection)
        
        # Define success criteria
        success_criteria = self._define_success_criteria(analysis)
        
        # Estimate duration
        estimated_duration = self._estimate_execution_duration(selection)
        
        execution_plan = ToolExecutionPlan(
            plan_id=plan_id,
            primary_strategy=primary_strategy,
            tool_sequence=tool_sequence,
            parallel_opportunities=parallel_opportunities,
            checkpoints=checkpoints,
            fallback_strategies=fallback_strategies,
            success_criteria=success_criteria,
            estimated_duration=estimated_duration
        )
        
        logger.info("execution_plan_created", 
                   plan_id=plan_id,
                   primary_strategy=primary_strategy,
                   estimated_duration=estimated_duration)
        
        return execution_plan
    
    def _assess_tool_capabilities(self, analysis: ComprehensiveQueryAnalysis) -> Dict[str, float]:
        """Assess each tool's capability to handle the query"""
        assessments = {}
        
        for tool_name, tool in self.available_tools.items():
            capability_score = tool.can_handle_query(analysis)
            assessments[tool_name] = capability_score
            logger.debug("tool_capability_assessed", 
                        tool_name=tool_name, 
                        capability_score=capability_score)
        
        return assessments
    
    def _apply_performance_learning(self, assessments: Dict[str, float], 
                                  analysis: ComprehensiveQueryAnalysis) -> Dict[str, float]:
        """Apply learned performance data to adjust tool scores"""
        adjusted = assessments.copy()
        
        # Apply performance history
        for tool_name in adjusted:
            if tool_name in self.performance_tracking:
                performance_data = self.performance_tracking[tool_name]
                
                # Factor in success rate
                success_rate = performance_data.get("success_rate", 1.0)
                adjusted[tool_name] *= success_rate
                
                # Factor in query type performance
                query_type = analysis.information_seeking.information_type
                if query_type in performance_data.get("query_type_performance", {}):
                    type_performance = performance_data["query_type_performance"][query_type]
                    adjusted[tool_name] *= type_performance
        
        logger.debug("performance_learning_applied", adjusted_scores=adjusted)
        return adjusted
    
    def _llm_assisted_selection(self, analysis: ComprehensiveQueryAnalysis, 
                              assessments: Dict[str, float]) -> str:
        """Use LLM to provide reasoning for tool selection"""
        
        reasoning_template = """
        Based on this query analysis and tool capability assessments, provide reasoning 
        for the best tool selection strategy.
        
        Query Analysis:
        - Complexity: {complexity}
        - Information Type: {info_type}
        - Requires Relationships: {requires_relationships}
        - Has Multiple Questions: {has_multiple_questions}
        - Medical Domain: {medical_domain}
        
        Tool Capability Scores:
        {tool_scores}
        
        Available Tools:
        - vector_search: Fast semantic similarity, good for factual queries
        - graph_search: Relationship analysis, complex reasoning
        - hybrid_search: Combined approach, comprehensive but resource-intensive
        
        Provide your reasoning for tool selection considering:
        1. Query requirements vs tool strengths
        2. Efficiency vs comprehensiveness trade-offs
        3. Resource utilization
        4. Expected accuracy
        
        Recommend primary tool selection and reasoning:
        """
        
        tool_scores_text = "\n".join([f"- {tool}: {score:.2f}" for tool, score in assessments.items()])
        
        try:
            reasoning_response = secure_llm_interaction(
                self.llm,
                reasoning_template,
                "",  # No user input needed here
                complexity=analysis.complexity.complexity_level,
                info_type=analysis.information_seeking.information_type,
                requires_relationships=analysis.information_seeking.requires_relationships,
                has_multiple_questions=analysis.sub_questions.has_multiple_questions,
                medical_domain=analysis.medical_domain.medical_domain,
                tool_scores=tool_scores_text
            )
            
            return reasoning_response
            
        except Exception as e:
            logger.error("llm_assisted_selection_failed", error=str(e))
            return "LLM reasoning failed, using capability scores for selection."
    
    def _create_selection_reasoning(self, analysis: ComprehensiveQueryAnalysis,
                                  assessments: Dict[str, float],
                                  llm_reasoning: str) -> ToolSelectionReasoning:
        """Create comprehensive tool selection reasoning"""
        
        # Select tools based on assessments
        sorted_tools = sorted(assessments.items(), key=lambda x: x[1], reverse=True)
        
        # Primary tool selection logic
        selected_tools = []
        reasoning_steps = []
        
        # If hybrid scores highest and complexity warrants it
        if (sorted_tools[0][0] == "hybrid_search" and 
            sorted_tools[0][1] > 0.8 and 
            analysis.complexity.complexity_level in ["Complex", "Multi-faceted"]):
            selected_tools = ["hybrid_search"]
            reasoning_steps.append("Selected hybrid search for complex, multi-faceted query")
        
        # If query requires relationships, prefer graph
        elif (analysis.information_seeking.requires_relationships and 
              assessments.get("graph_search", 0) > 0.6):
            selected_tools = ["graph_search"]
            reasoning_steps.append("Selected graph search for relationship-focused query")
        
        # For simple factual queries, prefer vector
        elif (analysis.information_seeking.information_type == "Factual" and 
              analysis.complexity.complexity_level == "Simple"):
            selected_tools = ["vector_search"]
            reasoning_steps.append("Selected vector search for simple factual query")
        
        # Default to top scoring tool
        else:
            selected_tools = [sorted_tools[0][0]]
            reasoning_steps.append(f"Selected {sorted_tools[0][0]} based on highest capability score")
        
        # Add backup tool
        if len(sorted_tools) > 1 and sorted_tools[1][1] > 0.5:
            backup_tool = sorted_tools[1][0]
            reasoning_steps.append(f"Added {backup_tool} as backup option")
        
        # Tool execution order
        tool_order = selected_tools.copy()
        
        # Confidence scores
        confidence_scores = {tool: assessments.get(tool, 0.0) for tool in selected_tools}
        
        # Contingency plans
        contingency_plans = [
            f"If primary tool fails, retry with {sorted_tools[1][0]}" if len(sorted_tools) > 1 else "Re-analyze query for alternative approach",
            "If all tools fail, provide general medical guidance",
            "If non-medical query detected, redirect appropriately"
        ]
        
        # Resource allocation
        resource_allocation = {}
        for tool in selected_tools:
            if tool == "hybrid_search":
                resource_allocation[tool] = "High"
            elif tool == "graph_search":
                resource_allocation[tool] = "Medium-High"
            else:
                resource_allocation[tool] = "Medium"
        
        # Expected outcomes
        expected_outcomes = {}
        for tool in selected_tools:
            if tool == "vector_search":
                expected_outcomes[tool] = "Relevant documents with semantic similarity"
            elif tool == "graph_search":
                expected_outcomes[tool] = "Relationship insights and connected entities"
            else:
                expected_outcomes[tool] = "Comprehensive analysis with multiple perspectives"
        
        reasoning_steps.append(f"LLM Reasoning: {llm_reasoning[:200]}...")
        
        return ToolSelectionReasoning(
            selected_tools=selected_tools,
            tool_order=tool_order,
            reasoning_steps=reasoning_steps,
            confidence_scores=confidence_scores,
            contingency_plans=contingency_plans,
            resource_allocation=resource_allocation,
            expected_outcomes=expected_outcomes
        )
    
    def _determine_execution_strategy(self, analysis: ComprehensiveQueryAnalysis, 
                                    selection: ToolSelectionReasoning) -> str:
        """Determine the primary execution strategy"""
        
        if len(selection.selected_tools) == 1:
            return f"Single-tool execution with {selection.selected_tools[0]}"
        elif "hybrid_search" in selection.selected_tools:
            return "Comprehensive hybrid approach with result synthesis"
        elif analysis.sub_questions.has_multiple_questions:
            return "Sequential processing of sub-questions"
        else:
            return "Multi-tool validation and comparison"
    
    def _create_tool_sequence(self, selection: ToolSelectionReasoning) -> List[Dict[str, Any]]:
        """Create ordered sequence of tool executions"""
        sequence = []
        
        for i, tool in enumerate(selection.tool_order):
            step = {
                "step": i + 1,
                "tool": tool,
                "purpose": selection.expected_outcomes.get(tool, "Process query"),
                "confidence": selection.confidence_scores.get(tool, 0.0),
                "resource_level": selection.resource_allocation.get(tool, "Medium")
            }
            sequence.append(step)
        
        return sequence
    
    def _identify_parallel_opportunities(self, selection: ToolSelectionReasoning) -> List[List[str]]:
        """Identify tools that can run in parallel"""
        parallel_groups = []
        
        # Vector and graph can often run in parallel
        if "vector_search" in selection.selected_tools and "graph_search" in selection.selected_tools:
            parallel_groups.append(["vector_search", "graph_search"])
        
        return parallel_groups
    
    def _define_execution_checkpoints(self, analysis: ComprehensiveQueryAnalysis, 
                                    selection: ToolSelectionReasoning) -> List[str]:
        """Define validation checkpoints during execution"""
        checkpoints = [
            "Validate medical relevance before processing",
            "Check tool execution success",
            "Validate result quality and completeness"
        ]
        
        if analysis.sub_questions.has_multiple_questions:
            checkpoints.insert(1, "Validate sub-question processing progress")
        
        if len(selection.selected_tools) > 1:
            checkpoints.append("Compare and synthesize results from multiple tools")
        
        return checkpoints
    
    def _create_fallback_strategies(self, analysis: ComprehensiveQueryAnalysis,
                                  selection: ToolSelectionReasoning) -> List[Dict[str, Any]]:
        """Create fallback strategies if primary tools fail"""
        strategies = []
        
        # Primary tool failure fallback
        if len(selection.selected_tools) > 1:
            strategies.append({
                "scenario": "Primary tool failure",
                "action": f"Switch to {selection.selected_tools[1]}",
                "trigger": "Tool execution error or timeout"
            })
        
        # Quality failure fallback
        strategies.append({
            "scenario": "Low quality results",
            "action": "Re-analyze query with different parameters",
            "trigger": "Result quality score below threshold"
        })
        
        # Complete failure fallback
        strategies.append({
            "scenario": "All tools fail",
            "action": "Provide general guidance based on query intent",
            "trigger": "No tools produce usable results"
        })
        
        return strategies
    
    def _define_success_criteria(self, analysis: ComprehensiveQueryAnalysis) -> List[str]:
        """Define criteria for measuring execution success"""
        criteria = [
            "Tool execution completes without errors",
            "Results are relevant to the query",
            "Medical validation passes if applicable"
        ]
        
        if analysis.information_seeking.information_type == "Comparative":
            criteria.append("Multiple perspectives are provided for comparison")
        
        if analysis.sub_questions.has_multiple_questions:
            criteria.append("All sub-questions are addressed")
        
        criteria.append("Result confidence score above minimum threshold")
        
        return criteria
    
    def _estimate_execution_duration(self, selection: ToolSelectionReasoning) -> str:
        """Estimate total execution duration"""
        if "hybrid_search" in selection.selected_tools:
            return "Extended (30-60 seconds)"
        elif len(selection.selected_tools) > 1:
            return "Standard (15-30 seconds)"
        elif "graph_search" in selection.selected_tools:
            return "Standard (10-20 seconds)"
        else:
            return "Fast (5-15 seconds)"
    
    def _create_fallback_selection(self, analysis: ComprehensiveQueryAnalysis) -> ToolSelectionReasoning:
        """Create fallback selection if reasoning fails"""
        
        # Simple fallback logic
        if analysis.information_seeking.requires_relationships:
            selected_tools = ["graph_search"]
        else:
            selected_tools = ["vector_search"]
        
        return ToolSelectionReasoning(
            selected_tools=selected_tools,
            tool_order=selected_tools,
            reasoning_steps=["Fallback selection due to reasoning failure"],
            confidence_scores={tool: 0.5 for tool in selected_tools},
            contingency_plans=["Manual query analysis if this fails"],
            resource_allocation={tool: "Medium" for tool in selected_tools},
            expected_outcomes={tool: "Basic query processing" for tool in selected_tools}
        )
    
    def _record_selection_decision(self, analysis: ComprehensiveQueryAnalysis,
                                 selection: ToolSelectionReasoning):
        """Record selection decision for learning"""
        record = {
            "timestamp": datetime.now(),
            "analysis_id": analysis.query_id,
            "query_type": analysis.information_seeking.information_type,
            "complexity": analysis.complexity.complexity_level,
            "selected_tools": selection.selected_tools,
            "confidence_scores": selection.confidence_scores
        }
        
        self.selection_history.append(record)
        
        # Keep only recent history
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]
        
        logger.debug("selection_decision_recorded", 
                    analysis_id=analysis.query_id,
                    selected_tools=selection.selected_tools)
    
    def update_tool_performance(self, tool_name: str, query_type: str, 
                              success: bool, quality_score: float, duration: float):
        """Update tool performance tracking"""
        if tool_name not in self.performance_tracking:
            self.performance_tracking[tool_name] = {
                "success_rate": 1.0,
                "average_quality": 1.0,
                "average_duration": 0.0,
                "query_type_performance": {}
            }
        
        perf = self.performance_tracking[tool_name]
        
        # Update tool performance
        if tool_name in self.available_tools:
            self.available_tools[tool_name].update_performance_metrics(success, duration)
            
        # Update query type specific performance
        if query_type not in perf["query_type_performance"]:
            perf["query_type_performance"][query_type] = quality_score
        else:
            # Running average
            current = perf["query_type_performance"][query_type]
            perf["query_type_performance"][query_type] = (current + quality_score) / 2
        
        logger.debug("tool_performance_updated", 
                    tool_name=tool_name,
                    query_type=query_type,
                    success=success,
                    quality_score=quality_score)
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about tool selections"""
        if not self.selection_history:
            return {"total_selections": 0}
        
        total = len(self.selection_history)
        tool_usage = {}
        complexity_patterns = {}
        
        for record in self.selection_history:
            # Count tool usage
            for tool in record["selected_tools"]:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
            
            # Count complexity patterns
            complexity = record["complexity"]
            complexity_patterns[complexity] = complexity_patterns.get(complexity, 0) + 1
        
        return {
            "total_selections": total,
            "tool_usage_distribution": tool_usage,
            "complexity_patterns": complexity_patterns,
            "most_used_tool": max(tool_usage.items(), key=lambda x: x[1])[0] if tool_usage else "none",
            "performance_tracking": self.performance_tracking
        }
