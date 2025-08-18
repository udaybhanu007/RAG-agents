"""
Execution Planner with Dynamic Planning and Contingencies

This module creates comprehensive execution plans based on query analysis and tool selection,
with built-in contingencies and adaptive execution strategies.
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic.v1 import BaseModel, Field
import asyncio
import time

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from updated_agents.enhanced_query_analyzer import ComprehensiveQueryAnalysis
from updated_agents.dynamic_tool_selector import ToolSelectionReasoning, ToolExecutionPlan
from core.input_sanitization import secure_llm_interaction
from core.logging_config import get_logger
from core.observability import traceable, get_traceable_config

logger = get_logger("execution_planner")

class ExecutionStatus(Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"

class ExecutionStep(BaseModel):
    """Individual execution step"""
    step_id: str = Field(description="Unique step identifier")
    step_name: str = Field(description="Human-readable step name")
    tool_name: str = Field(description="Tool to execute")
    parameters: Dict[str, Any] = Field(description="Execution parameters")
    dependencies: List[str] = Field(description="Steps that must complete first")
    timeout_seconds: int = Field(description="Maximum execution time")
    retry_count: int = Field(description="Number of retry attempts")
    success_criteria: List[str] = Field(description="Criteria for success")
    fallback_action: Optional[str] = Field(description="Action if step fails")

class ContingencyAction(BaseModel):
    """Contingency action definition"""
    trigger_condition: str = Field(description="Condition that triggers this action")
    action_type: str = Field(description="Type of action (retry, fallback, escalate)")
    action_details: Dict[str, Any] = Field(description="Specific action details")
    priority: int = Field(description="Priority level (1-10)")
    max_attempts: int = Field(description="Maximum attempts for this action")

class ExecutionContext(BaseModel):
    """Context for execution tracking"""
    execution_id: str = Field(description="Unique execution identifier")
    start_time: str = Field(description="Execution start timestamp")
    query_analysis: ComprehensiveQueryAnalysis
    tool_selection: ToolSelectionReasoning
    execution_plan: ToolExecutionPlan
    current_step: int = Field(description="Current step index")
    completed_steps: List[str] = Field(description="Completed step IDs")
    failed_steps: List[str] = Field(description="Failed step IDs")
    execution_results: Dict[str, Any] = Field(description="Results from each step")
    contingency_actions_taken: List[str] = Field(description="Contingency actions executed")
    quality_scores: Dict[str, float] = Field(description="Quality scores for each step")

class ExecutionResult(BaseModel):
    """Final execution result"""
    execution_id: str = Field(description="Execution identifier")
    status: ExecutionStatus = Field(description="Final execution status")
    total_duration: float = Field(description="Total execution time in seconds")
    steps_completed: int = Field(description="Number of steps completed")
    steps_failed: int = Field(description="Number of steps failed")
    final_answer: str = Field(description="Final synthesized answer")
    sources: List[str] = Field(description="Source documents used")
    confidence_score: float = Field(description="Overall confidence in result")
    quality_metrics: Dict[str, float] = Field(description="Quality metrics")
    lessons_learned: List[str] = Field(description="Insights for future executions")

class ExecutionPlanner:
    """
    Dynamic Execution Planner with Contingencies
    
    This class creates and manages execution plans with built-in contingencies,
    adaptive strategies, and learning from execution outcomes.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.execution_history = []
        self.performance_patterns = {}
        self.contingency_library = {}
        self._init_contingency_library()
        logger.info("execution_planner_initialized")
    
    def _init_contingency_library(self):
        """Initialize library of contingency actions"""
        self.contingency_library = {
            "tool_timeout": ContingencyAction(
                trigger_condition="Tool execution exceeds timeout",
                action_type="retry",
                action_details={"retry_with_reduced_scope": True, "timeout_multiplier": 1.5},
                priority=8,
                max_attempts=2
            ),
            "low_quality_results": ContingencyAction(
                trigger_condition="Result quality below threshold",
                action_type="fallback",
                action_details={"try_alternative_tool": True, "quality_threshold": 0.6},
                priority=7,
                max_attempts=1
            ),
            "tool_execution_error": ContingencyAction(
                trigger_condition="Tool execution throws error",
                action_type="retry",
                action_details={"retry_with_sanitized_params": True, "wait_seconds": 5},
                priority=9,
                max_attempts=3
            ),
            "insufficient_results": ContingencyAction(
                trigger_condition="Tool returns no or minimal results",
                action_type="escalate",
                action_details={"try_broader_search": True, "expand_query_scope": True},
                priority=6,
                max_attempts=1
            ),
            "partial_failure": ContingencyAction(
                trigger_condition="Some steps succeed, others fail",
                action_type="fallback",
                action_details={"synthesize_partial_results": True, "note_limitations": True},
                priority=5,
                max_attempts=1
            )
        }
    
    @traceable(**get_traceable_config("ExecutionPlanner"))
    def create_comprehensive_plan(self, analysis: ComprehensiveQueryAnalysis,
                                selection: ToolSelectionReasoning,
                                base_plan: ToolExecutionPlan) -> ToolExecutionPlan:
        """
        Create comprehensive execution plan with contingencies
        
        Args:
            analysis: Query analysis
            selection: Tool selection reasoning
            base_plan: Base execution plan
            
        Returns:
            Enhanced execution plan with contingencies
        """
        logger.info("comprehensive_plan_creation_started",
                   analysis_id=analysis.query_id,
                   base_plan_id=base_plan.plan_id)
        
        try:
            # Create detailed execution steps
            execution_steps = self._create_detailed_steps(analysis, selection, base_plan)
            
            # Add contingency planning
            contingency_strategies = self._plan_contingencies(analysis, selection, execution_steps)
            
            # Create monitoring checkpoints
            monitoring_checkpoints = self._create_monitoring_checkpoints(execution_steps)
            
            # Plan resource allocation
            resource_plan = self._plan_resource_allocation(execution_steps)
            
            # Create enhanced execution plan
            enhanced_plan = ToolExecutionPlan(
                plan_id=f"enhanced_{base_plan.plan_id}",
                primary_strategy=base_plan.primary_strategy,
                tool_sequence=base_plan.tool_sequence + [{"execution_steps": execution_steps}],
                parallel_opportunities=base_plan.parallel_opportunities,
                checkpoints=base_plan.checkpoints + monitoring_checkpoints,
                fallback_strategies=base_plan.fallback_strategies + contingency_strategies,
                success_criteria=base_plan.success_criteria,
                estimated_duration=self._calculate_enhanced_duration(execution_steps)
            )
            
            logger.info("comprehensive_plan_created",
                       enhanced_plan_id=enhanced_plan.plan_id,
                       total_steps=len(execution_steps),
                       contingency_count=len(contingency_strategies))
            
            return enhanced_plan
            
        except Exception as e:
            logger.error("comprehensive_plan_creation_failed", error=str(e))
            # Return base plan if enhancement fails
            return base_plan
    
    @traceable(**get_traceable_config("ExecutionPlanner"))
    async def execute_plan_with_contingencies(self, analysis: ComprehensiveQueryAnalysis,
                                            selection: ToolSelectionReasoning,
                                            execution_plan: ToolExecutionPlan,
                                            tool_executors: Dict[str, Callable]) -> ExecutionResult:
        """
        Execute plan with contingency handling and adaptive execution
        
        Args:
            analysis: Query analysis
            selection: Tool selection reasoning
            execution_plan: Execution plan
            tool_executors: Dictionary of tool executor functions
            
        Returns:
            ExecutionResult: Complete execution result with metrics
        """
        execution_id = f"exec_{analysis.query_id}_{datetime.now().strftime('%H%M%S')}"
        start_time = datetime.now()
        
        logger.info("plan_execution_started",
                   execution_id=execution_id,
                   plan_id=execution_plan.plan_id)
        
        # Initialize execution context
        context = ExecutionContext(
            execution_id=execution_id,
            start_time=start_time.isoformat(),
            query_analysis=analysis,
            tool_selection=selection,
            execution_plan=execution_plan,
            current_step=0,
            completed_steps=[],
            failed_steps=[],
            execution_results={},
            contingency_actions_taken=[],
            quality_scores={}
        )
        
        try:
            # Execute steps with contingency handling
            await self._execute_steps_with_monitoring(context, tool_executors)
            
            # Synthesize final results
            final_result = await self._synthesize_execution_results(context)
            
            # Learn from execution
            self._learn_from_execution(context, final_result)
            
            logger.info("plan_execution_completed",
                       execution_id=execution_id,
                       status=final_result.status,
                       duration=final_result.total_duration)
            
            return final_result
            
        except Exception as e:
            logger.error("plan_execution_failed", 
                        execution_id=execution_id,
                        error=str(e))
            
            # Create failure result
            return self._create_failure_result(context, str(e))
    
    def _create_detailed_steps(self, analysis: ComprehensiveQueryAnalysis,
                             selection: ToolSelectionReasoning,
                             base_plan: ToolExecutionPlan) -> List[ExecutionStep]:
        """Create detailed execution steps"""
        steps = []
        
        # Step 1: Medical validation (if applicable)
        if analysis.medical_domain.is_medical:
            steps.append(ExecutionStep(
                step_id="medical_validation",
                step_name="Validate Medical Relevance",
                tool_name="medical_validator",
                parameters={"query": analysis.query_id, "confidence_threshold": 0.7},
                dependencies=[],
                timeout_seconds=10,
                retry_count=2,
                success_criteria=["Medical validation passes", "Confidence above threshold"],
                fallback_action="proceed_with_caution"
            ))
        
        # Step 2: Sub-question decomposition (if needed)
        if analysis.sub_questions.has_multiple_questions:
            steps.append(ExecutionStep(
                step_id="subquestion_decomposition",
                step_name="Decompose Sub-questions",
                tool_name="query_decomposer",
                parameters={"sub_questions": analysis.sub_questions.sub_questions},
                dependencies=["medical_validation"] if analysis.medical_domain.is_medical else [],
                timeout_seconds=15,
                retry_count=1,
                success_criteria=["Sub-questions identified", "Processing order determined"],
                fallback_action="process_as_single_query"
            ))
        
        # Step 3: Primary tool execution
        for i, tool in enumerate(selection.selected_tools):
            step_dependencies = []
            if i == 0:
                if analysis.medical_domain.is_medical:
                    step_dependencies.append("medical_validation")
                if analysis.sub_questions.has_multiple_questions:
                    step_dependencies.append("subquestion_decomposition")
            else:
                step_dependencies.append(f"tool_execution_{i-1}")
            
            steps.append(ExecutionStep(
                step_id=f"tool_execution_{i}",
                step_name=f"Execute {tool}",
                tool_name=tool,
                parameters=self._get_tool_parameters(tool, analysis),
                dependencies=step_dependencies,
                timeout_seconds=self._get_tool_timeout(tool),
                retry_count=self._get_tool_retry_count(tool),
                success_criteria=["Tool executes successfully", "Results meet quality threshold"],
                fallback_action=self._get_tool_fallback(tool, selection)
            ))
        
        # Step 4: Result synthesis
        steps.append(ExecutionStep(
            step_id="result_synthesis",
            step_name="Synthesize Results",
            tool_name="result_synthesizer",
            parameters={"synthesis_strategy": "comprehensive"},
            dependencies=[f"tool_execution_{i}" for i in range(len(selection.selected_tools))],
            timeout_seconds=20,
            retry_count=2,
            success_criteria=["Results synthesized", "Answer quality acceptable"],
            fallback_action="partial_synthesis"
        ))
        
        # Step 5: Quality validation
        steps.append(ExecutionStep(
            step_id="quality_validation",
            step_name="Validate Answer Quality",
            tool_name="quality_validator",
            parameters={"quality_threshold": 0.6},
            dependencies=["result_synthesis"],
            timeout_seconds=10,
            retry_count=1,
            success_criteria=["Quality validation passes"],
            fallback_action="accept_with_warnings"
        ))
        
        return steps
    
    def _plan_contingencies(self, analysis: ComprehensiveQueryAnalysis,
                          selection: ToolSelectionReasoning,
                          steps: List[ExecutionStep]) -> List[Dict[str, Any]]:
        """Plan contingency strategies for execution"""
        contingencies = []
        
        # Tool-specific contingencies
        for step in steps:
            if step.tool_name in ["vector_search", "graph_search", "hybrid_search"]:
                contingencies.append({
                    "scenario": f"{step.tool_name}_timeout",
                    "action": "Retry with reduced scope and increased timeout",
                    "trigger": f"Step {step.step_id} exceeds timeout",
                    "priority": 8
                })
                
                contingencies.append({
                    "scenario": f"{step.tool_name}_quality_failure",
                    "action": "Try alternative tool or broaden search parameters",
                    "trigger": f"Step {step.step_id} produces low quality results",
                    "priority": 7
                })
        
        # Query complexity contingencies
        if analysis.complexity.complexity_level in ["Complex", "Multi-faceted"]:
            contingencies.append({
                "scenario": "complex_query_partial_failure",
                "action": "Break down into simpler sub-components and retry",
                "trigger": "Multiple steps fail on complex query",
                "priority": 6
            })
        
        # Medical domain contingencies
        if analysis.medical_domain.is_medical:
            contingencies.append({
                "scenario": "medical_validation_failure",
                "action": "Provide general medical guidance disclaimer",
                "trigger": "Medical validation step fails",
                "priority": 9
            })
        
        # Multi-tool contingencies
        if len(selection.selected_tools) > 1:
            contingencies.append({
                "scenario": "tool_result_conflict",
                "action": "Use confidence scores to select best result",
                "trigger": "Tools produce conflicting results",
                "priority": 5
            })
        
        return contingencies
    
    def _create_monitoring_checkpoints(self, steps: List[ExecutionStep]) -> List[str]:
        """Create monitoring checkpoints for execution"""
        checkpoints = []
        
        # Checkpoint after each critical step
        critical_steps = ["medical_validation", "tool_execution_0", "result_synthesis"]
        for step_id in critical_steps:
            if any(s.step_id == step_id for s in steps):
                checkpoints.append(f"Monitor {step_id} completion and quality")
        
        # Resource utilization checkpoints
        checkpoints.append("Monitor resource utilization every 15 seconds")
        
        # Quality checkpoints
        checkpoints.append("Validate intermediate results quality")
        
        # Progress checkpoints
        checkpoints.append("Check execution progress against time estimates")
        
        return checkpoints
    
    def _plan_resource_allocation(self, steps: List[ExecutionStep]) -> Dict[str, Any]:
        """Plan resource allocation for execution"""
        return {
            "cpu_allocation": "dynamic_based_on_step",
            "memory_allocation": "conservative_with_buffers",
            "timeout_management": "adaptive_based_on_complexity",
            "parallel_execution": "when_steps_independent",
            "resource_monitoring": "continuous"
        }
    
    def _calculate_enhanced_duration(self, steps: List[ExecutionStep]) -> str:
        """Calculate estimated duration for enhanced plan"""
        total_seconds = sum(step.timeout_seconds for step in steps)
        
        # Add buffer for contingencies and monitoring
        total_seconds = int(total_seconds * 1.3)
        
        if total_seconds < 30:
            return "Fast (under 30 seconds)"
        elif total_seconds < 60:
            return "Standard (30-60 seconds)"
        elif total_seconds < 120:
            return "Extended (1-2 minutes)"
        else:
            return "Comprehensive (over 2 minutes)"
    
    async def _execute_steps_with_monitoring(self, context: ExecutionContext,
                                           tool_executors: Dict[str, Callable]):
        """Execute steps with real-time monitoring and contingency handling"""
        steps = self._extract_execution_steps(context.execution_plan)
        
        for i, step in enumerate(steps):
            context.current_step = i
            logger.info("executing_step", 
                       step_id=step.step_id,
                       step_name=step.step_name)
            
            try:
                # Check dependencies
                if not self._check_dependencies(step, context.completed_steps):
                    logger.warning("step_dependencies_not_met", step_id=step.step_id)
                    continue
                
                # Execute step with timeout
                step_start = time.time()
                step_result = await self._execute_step_with_timeout(
                    step, tool_executors, context
                )
                step_duration = time.time() - step_start
                
                # Validate step result
                quality_score = self._validate_step_result(step, step_result)
                context.quality_scores[step.step_id] = quality_score
                
                # Handle success
                if quality_score >= 0.6:  # Quality threshold
                    context.completed_steps.append(step.step_id)
                    context.execution_results[step.step_id] = step_result
                    logger.info("step_completed_successfully", 
                               step_id=step.step_id,
                               quality_score=quality_score)
                else:
                    # Handle quality failure
                    await self._handle_step_failure(step, context, "low_quality")
                
            except asyncio.TimeoutError:
                logger.warning("step_timeout", step_id=step.step_id)
                await self._handle_step_failure(step, context, "timeout")
                
            except Exception as e:
                logger.error("step_execution_error", 
                            step_id=step.step_id,
                            error=str(e))
                await self._handle_step_failure(step, context, "execution_error")
    
    async def _execute_step_with_timeout(self, step: ExecutionStep,
                                       tool_executors: Dict[str, Callable],
                                       context: ExecutionContext) -> Any:
        """Execute individual step with timeout"""
        if step.tool_name not in tool_executors:
            raise ValueError(f"Tool executor not found: {step.tool_name}")
        
        executor = tool_executors[step.tool_name]
        
        # Execute with timeout
        return await asyncio.wait_for(
            executor(step.parameters, context),
            timeout=step.timeout_seconds
        )
    
    def _validate_step_result(self, step: ExecutionStep, result: Any) -> float:
        """Validate step result and return quality score"""
        if result is None:
            return 0.0
        
        # Basic validation
        quality_score = 0.5
        
        # Check success criteria
        if hasattr(result, 'quality_metrics'):
            quality_score = result.quality_metrics.get('overall_quality', 0.5)
        elif isinstance(result, dict):
            if result.get('success', False):
                quality_score = 0.8
            if 'quality_score' in result:
                quality_score = result['quality_score']
        
        return max(0.0, min(1.0, quality_score))
    
    async def _handle_step_failure(self, step: ExecutionStep, context: ExecutionContext,
                                 failure_type: str):
        """Handle step failure with contingency actions"""
        logger.warning("handling_step_failure", 
                      step_id=step.step_id,
                      failure_type=failure_type)
        
        # Find applicable contingency
        contingency = self._find_applicable_contingency(failure_type, step)
        
        if contingency:
            action_taken = await self._execute_contingency_action(
                contingency, step, context
            )
            context.contingency_actions_taken.append(action_taken)
        else:
            # Default fallback
            context.failed_steps.append(step.step_id)
            if step.fallback_action:
                logger.info("executing_fallback_action",
                           step_id=step.step_id,
                           fallback_action=step.fallback_action)
    
    def _find_applicable_contingency(self, failure_type: str, 
                                   step: ExecutionStep) -> Optional[ContingencyAction]:
        """Find applicable contingency action for failure"""
        contingency_map = {
            "timeout": "tool_timeout",
            "low_quality": "low_quality_results",
            "execution_error": "tool_execution_error"
        }
        
        contingency_key = contingency_map.get(failure_type)
        return self.contingency_library.get(contingency_key)
    
    async def _execute_contingency_action(self, contingency: ContingencyAction,
                                        step: ExecutionStep,
                                        context: ExecutionContext) -> str:
        """Execute contingency action"""
        action_details = contingency.action_details
        action_type = contingency.action_type
        
        logger.info("executing_contingency_action",
                   action_type=action_type,
                   step_id=step.step_id)
        
        if action_type == "retry":
            # Implement retry logic
            return f"Retried {step.step_id} with modified parameters"
        
        elif action_type == "fallback":
            # Implement fallback logic
            return f"Applied fallback strategy for {step.step_id}"
        
        elif action_type == "escalate":
            # Implement escalation logic
            return f"Escalated {step.step_id} to alternative approach"
        
        return f"Applied {action_type} action to {step.step_id}"
    
    async def _synthesize_execution_results(self, context: ExecutionContext) -> ExecutionResult:
        """Synthesize final execution results"""
        end_time = datetime.now()
        start_time = datetime.fromisoformat(context.start_time)
        total_duration = (end_time - start_time).total_seconds()
        
        # Determine overall status
        if len(context.failed_steps) == 0:
            status = ExecutionStatus.COMPLETED
        elif len(context.completed_steps) > len(context.failed_steps):
            status = ExecutionStatus.COMPLETED  # Partial success
        else:
            status = ExecutionStatus.FAILED
        
        # Synthesize final answer
        final_answer = self._synthesize_final_answer(context)
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(context)
        
        # Extract sources
        sources = self._extract_sources(context)
        
        # Generate quality metrics
        quality_metrics = self._calculate_quality_metrics(context)
        
        # Generate lessons learned
        lessons_learned = self._extract_lessons_learned(context)
        
        return ExecutionResult(
            execution_id=context.execution_id,
            status=status,
            total_duration=total_duration,
            steps_completed=len(context.completed_steps),
            steps_failed=len(context.failed_steps),
            final_answer=final_answer,
            sources=sources,
            confidence_score=confidence_score,
            quality_metrics=quality_metrics,
            lessons_learned=lessons_learned
        )
    
    def _synthesize_final_answer(self, context: ExecutionContext) -> str:
        """Synthesize final answer from execution results"""
        if not context.execution_results:
            return "Unable to generate answer due to execution failures."
        
        # Simple synthesis - in production would be more sophisticated
        answers = []
        for step_id, result in context.execution_results.items():
            if isinstance(result, dict) and 'answer' in result:
                answers.append(result['answer'])
            elif isinstance(result, str):
                answers.append(result)
        
        if answers:
            return " ".join(answers)
        else:
            return "Execution completed but no clear answer could be synthesized."
    
    def _calculate_overall_confidence(self, context: ExecutionContext) -> float:
        """Calculate overall confidence score"""
        if not context.quality_scores:
            return 0.0
        
        return sum(context.quality_scores.values()) / len(context.quality_scores)
    
    def _extract_sources(self, context: ExecutionContext) -> List[str]:
        """Extract source documents from execution results"""
        sources = []
        for result in context.execution_results.values():
            if isinstance(result, dict) and 'sources' in result:
                sources.extend(result['sources'])
        
        return list(set(sources))  # Remove duplicates
    
    def _calculate_quality_metrics(self, context: ExecutionContext) -> Dict[str, float]:
        """Calculate quality metrics"""
        return {
            "step_success_rate": len(context.completed_steps) / max(1, len(context.completed_steps) + len(context.failed_steps)),
            "average_quality": sum(context.quality_scores.values()) / max(1, len(context.quality_scores)),
            "contingency_rate": len(context.contingency_actions_taken) / max(1, len(context.completed_steps) + len(context.failed_steps))
        }
    
    def _extract_lessons_learned(self, context: ExecutionContext) -> List[str]:
        """Extract lessons learned from execution"""
        lessons = []
        
        if context.failed_steps:
            lessons.append(f"Failed steps: {', '.join(context.failed_steps)} - investigate failure patterns")
        
        if context.contingency_actions_taken:
            lessons.append(f"Contingencies used: {len(context.contingency_actions_taken)} - review trigger conditions")
        
        if all(score > 0.8 for score in context.quality_scores.values()):
            lessons.append("High quality execution - current strategy is effective")
        
        return lessons
    
    def _learn_from_execution(self, context: ExecutionContext, result: ExecutionResult):
        """Learn from execution outcomes"""
        execution_record = {
            "timestamp": datetime.now(),
            "execution_id": context.execution_id,
            "query_type": context.query_analysis.information_seeking.information_type,
            "complexity": context.query_analysis.complexity.complexity_level,
            "tools_used": context.tool_selection.selected_tools,
            "success_rate": result.quality_metrics.get("step_success_rate", 0.0),
            "overall_confidence": result.confidence_score,
            "contingencies_used": len(context.contingency_actions_taken),
            "lessons_learned": result.lessons_learned
        }
        
        self.execution_history.append(execution_record)
        
        # Update performance patterns
        query_type = context.query_analysis.information_seeking.information_type
        if query_type not in self.performance_patterns:
            self.performance_patterns[query_type] = []
        
        self.performance_patterns[query_type].append({
            "success_rate": result.quality_metrics.get("step_success_rate", 0.0),
            "confidence": result.confidence_score,
            "tools_used": context.tool_selection.selected_tools
        })
        
        # Keep only recent patterns
        if len(self.performance_patterns[query_type]) > 20:
            self.performance_patterns[query_type] = self.performance_patterns[query_type][-20:]
        
        logger.info("execution_learning_completed",
                   execution_id=context.execution_id,
                   success_rate=result.quality_metrics.get("step_success_rate", 0.0))
    
    def _create_failure_result(self, context: ExecutionContext, error: str) -> ExecutionResult:
        """Create result for failed execution"""
        return ExecutionResult(
            execution_id=context.execution_id,
            status=ExecutionStatus.FAILED,
            total_duration=0.0,
            steps_completed=0,
            steps_failed=1,
            final_answer=f"Execution failed: {error}",
            sources=[],
            confidence_score=0.0,
            quality_metrics={"step_success_rate": 0.0, "average_quality": 0.0},
            lessons_learned=[f"Execution failed with error: {error}"]
        )
    
    def _extract_execution_steps(self, plan: ToolExecutionPlan) -> List[ExecutionStep]:
        """Extract execution steps from plan"""
        # This would extract steps from the enhanced plan
        # For now, return empty list - would be implemented based on actual plan structure
        return []
    
    def _check_dependencies(self, step: ExecutionStep, completed_steps: List[str]) -> bool:
        """Check if step dependencies are satisfied"""
        return all(dep in completed_steps for dep in step.dependencies)
    
    def _get_tool_parameters(self, tool: str, analysis: ComprehensiveQueryAnalysis) -> Dict[str, Any]:
        """Get parameters for tool execution"""
        base_params = {"query_analysis": analysis.query_id}
        
        if tool == "vector_search":
            base_params.update({"similarity_threshold": 0.7, "max_results": 10})
        elif tool == "graph_search":
            base_params.update({"max_depth": 3, "relationship_types": ["all"]})
        elif tool == "hybrid_search":
            base_params.update({"vector_weight": 0.6, "graph_weight": 0.4})
        
        return base_params
    
    def _get_tool_timeout(self, tool: str) -> int:
        """Get timeout for tool"""
        timeouts = {
            "vector_search": 15,
            "graph_search": 25,
            "hybrid_search": 40,
            "medical_validator": 10,
            "result_synthesizer": 20
        }
        return timeouts.get(tool, 30)
    
    def _get_tool_retry_count(self, tool: str) -> int:
        """Get retry count for tool"""
        return 2 if tool in ["vector_search", "graph_search"] else 1
    
    def _get_tool_fallback(self, tool: str, selection: ToolSelectionReasoning) -> str:
        """Get fallback action for tool"""
        fallbacks = {
            "vector_search": "try_graph_search",
            "graph_search": "try_vector_search",
            "hybrid_search": "try_best_individual_tool"
        }
        return fallbacks.get(tool, "proceed_with_partial_results")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for learning"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total = len(self.execution_history)
        avg_success = sum(e["success_rate"] for e in self.execution_history) / total
        avg_confidence = sum(e["overall_confidence"] for e in self.execution_history) / total
        
        return {
            "total_executions": total,
            "average_success_rate": avg_success,
            "average_confidence": avg_confidence,
            "performance_patterns": self.performance_patterns,
            "most_effective_tools": self._get_most_effective_tools()
        }
    
    def _get_most_effective_tools(self) -> Dict[str, float]:
        """Get most effective tools based on execution history"""
        tool_performance = {}
        
        for execution in self.execution_history:
            for tool in execution["tools_used"]:
                if tool not in tool_performance:
                    tool_performance[tool] = []
                tool_performance[tool].append(execution["success_rate"])
        
        # Calculate average performance for each tool
        avg_performance = {}
        for tool, performances in tool_performance.items():
            avg_performance[tool] = sum(performances) / len(performances)
        
        return avg_performance
