"""
Enhanced Query Analyzer with Comprehensive LLM-based Analysis

This module provides deep analysis of user queries to understand:
1. Medical relevance and domain classification
2. Query complexity and type of information sought
3. Required tools and processing steps
4. Sub-questions and relationship requirements
"""

import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic.v1 import BaseModel, Field

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from core.input_sanitization import secure_llm_interaction, sanitize_user_input, detect_prompt_injection
from core.logging_config import get_logger
from core.observability import traceable, get_traceable_config

logger = get_logger("enhanced_query_analyzer")

class QueryComplexityAnalysis(BaseModel):
    """Detailed complexity analysis of the query"""
    complexity_level: str = Field(description="Simple, Moderate, Complex, or Multi-faceted")
    reasoning: str = Field(description="Explanation of complexity assessment")
    estimated_processing_time: str = Field(description="Estimated time category")
    requires_multiple_steps: bool = Field(description="Whether query needs multiple processing steps")

class MedicalDomainAnalysis(BaseModel):
    """Medical domain and relevance analysis"""
    is_medical: bool = Field(description="Whether query is medical/healthcare related")
    medical_domain: str = Field(description="Specific medical domain (radiology, cardiology, etc.)")
    confidence_score: float = Field(description="Confidence in medical classification (0-1)")
    medical_entities: List[str] = Field(description="Medical entities found in query")
    clinical_context: str = Field(description="Clinical context or scenario")

class InformationSeekingAnalysis(BaseModel):
    """Analysis of what type of information is being sought"""
    information_type: str = Field(description="Factual, Comparative, Analytical, Procedural, or Diagnostic")
    specific_needs: List[str] = Field(description="Specific information needs identified")
    requires_relationships: bool = Field(description="Whether query needs relationship data")
    requires_quantitative_data: bool = Field(description="Whether query needs numerical/statistical data")
    temporal_aspect: str = Field(description="Time-related aspects of the query")

class SubQuestionAnalysis(BaseModel):
    """Analysis of sub-questions within the main query"""
    has_multiple_questions: bool = Field(description="Whether query contains multiple sub-questions")
    sub_questions: List[str] = Field(description="List of identified sub-questions")
    question_dependencies: List[str] = Field(description="Dependencies between sub-questions")
    processing_order: List[str] = Field(description="Recommended order for processing sub-questions")

class ToolRequirementAnalysis(BaseModel):
    """Analysis of what tools would be most effective"""
    recommended_tools: List[str] = Field(description="List of recommended tools")
    tool_priorities: Dict[str, int] = Field(description="Priority ranking of tools (1-10)")
    tool_reasoning: Dict[str, str] = Field(description="Reasoning for each tool recommendation")
    fallback_options: List[str] = Field(description="Alternative tools if primary options fail")

class ComprehensiveQueryAnalysis(BaseModel):
    """Complete analysis of the user query"""
    query_id: str = Field(description="Unique identifier for this analysis")
    timestamp: str = Field(description="Analysis timestamp")
    complexity: QueryComplexityAnalysis
    medical_domain: MedicalDomainAnalysis
    information_seeking: InformationSeekingAnalysis
    sub_questions: SubQuestionAnalysis
    tool_requirements: ToolRequirementAnalysis
    overall_strategy: str = Field(description="Overall processing strategy recommendation")

class EnhancedQueryAnalyzer:
    """
    Enhanced Query Analyzer with comprehensive LLM-based analysis
    
    This analyzer provides deep understanding of user queries to enable
    intelligent tool selection and execution planning.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.analysis_count = 0
        self.analysis_history = []
        logger.info("enhanced_query_analyzer_initialized")
    
    @traceable(**get_traceable_config("EnhancedQueryAnalyzer"))
    def analyze_query_comprehensive(self, query: str, trace_id: Optional[str] = None) -> ComprehensiveQueryAnalysis:
        """
        Perform comprehensive analysis of the user query using optimized single LLM call
        
        Args:
            query: The user query to analyze
            trace_id: Optional trace ID for logging
            
        Returns:
            ComprehensiveQueryAnalysis: Complete structured analysis
        """
        logger.info("comprehensive_query_analysis_started", 
                   query_length=len(query),
                   trace_id=trace_id)
        
        # Security check - sanitize input and detect injection
        if detect_prompt_injection(query):
            logger.warning("prompt_injection_detected_in_query", trace_id=trace_id)
            raise ValueError("Potential prompt injection detected in query")
        
        sanitized_query = sanitize_user_input(query)
        
        # Generate unique analysis ID
        analysis_id = f"analysis_{self.analysis_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.analysis_count += 1
        
        try:
            # Single optimized LLM call for all analyses
            comprehensive_analysis = self._analyze_query_unified(sanitized_query, analysis_id, trace_id)
            
            # Store in history for learning (async to avoid blocking)
            self._store_analysis_history(analysis_id, sanitized_query, comprehensive_analysis, trace_id)
            
            logger.info("comprehensive_query_analysis_completed", 
                       analysis_id=analysis_id,
                       overall_strategy=comprehensive_analysis.overall_strategy,
                       trace_id=trace_id)
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error("comprehensive_query_analysis_failed", 
                        error=str(e),
                        trace_id=trace_id)
            # Re-raise the exception instead of fallback
            raise
    
    def _analyze_query_unified(self, query: str, analysis_id: str, trace_id: Optional[str] = None) -> ComprehensiveQueryAnalysis:
        """
        Unified analysis using a single LLM call for all analysis aspects
        
        Args:
            query: The sanitized query to analyze
            analysis_id: Unique analysis identifier
            trace_id: Optional trace ID for logging
            
        Returns:
            ComprehensiveQueryAnalysis: Complete structured analysis
        """
        logger.debug("unified_query_analysis_started", 
                    analysis_id=analysis_id, 
                    trace_id=trace_id)
        
        # Unified analysis template that captures all aspects in one call
        unified_template = """
        Analyze this query comprehensively and provide structured output for all analysis aspects.
        
        Query to analyze: {user_query}
        
        Provide your analysis in this EXACT format (each section on a new line):
        
        MEDICAL: [Yes/No]
        MEDICAL_DOMAIN: [Radiology/Cardiology/Oncology/General Medicine/Not Medical]
        MEDICAL_CONFIDENCE: [0.0-1.0]
        MEDICAL_ENTITIES: [comma-separated list of medical terms found]
        CLINICAL_CONTEXT: [brief description of clinical scenario]
        
        COMPLEXITY: [Simple/Moderate/Complex/Multi-faceted]
        COMPLEXITY_REASONING: [brief explanation of complexity assessment]
        PROCESSING_TIME: [Quick/Standard/Extended/Comprehensive]
        MULTIPLE_STEPS: [Yes/No]
        
        INFO_TYPE: [Factual/Comparative/Analytical/Procedural/Diagnostic]
        SPECIFIC_NEEDS: [comma-separated list of specific information requirements]
        RELATIONSHIPS_NEEDED: [Yes/No]
        QUANTITATIVE_DATA: [Yes/No]
        TEMPORAL_ASPECT: [Current/Historical/Trending/Predictive/Not Applicable]
        
        MULTIPLE_QUESTIONS: [Yes/No]
        SUB_QUESTIONS: [comma-separated list of sub-questions if any]
        QUESTION_DEPENDENCIES: [brief description of dependencies]
        PROCESSING_ORDER: [recommended order for processing]
        
        RECOMMENDED_TOOLS: [comma-separated list from: vector_search, graph_search, both_searches, medical_validation, query_analysis, document_retrieval, relationship_analysis, statistical_analysis]
        TOOL_PRIORITIES: [tool1:priority1, tool2:priority2, etc. where priority is 1-10]
        PRIMARY_TOOL_REASONING: [brief explanation for top recommended tool]
        FALLBACK_TOOLS: [comma-separated list of alternative tools]
        
        OVERALL_STRATEGY: [brief processing strategy recommendation]
        """
        
        try:
            response = secure_llm_interaction(
                self.llm, 
                unified_template, 
                query
            )
            
            # Parse the unified response into structured analysis
            return self._parse_unified_response(response, query, analysis_id, trace_id)
            
        except Exception as e:
            logger.error("unified_query_analysis_failed", 
                        error=str(e),
                        analysis_id=analysis_id,
                        trace_id=trace_id)
            # Re-raise the exception instead of fallback
            raise
    
    def _parse_unified_response(self, response: str, query: str, analysis_id: str, trace_id: Optional[str] = None) -> ComprehensiveQueryAnalysis:
        """
        Parse the unified LLM response into structured analysis components
        
        Args:
            response: Raw LLM response
            query: Original query
            analysis_id: Analysis identifier
            trace_id: Optional trace ID
            
        Returns:
            ComprehensiveQueryAnalysis: Parsed structured analysis
        """
        logger.debug("parsing_unified_response", 
                    analysis_id=analysis_id, 
                    trace_id=trace_id)
        
        lines = response.strip().split('\n')
        parsed_data = {}
        
        # Parse response into key-value pairs
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                parsed_data[key.strip().upper()] = value.strip()
        
        # Helper function to safely get values with defaults
        def get_value(key: str, default: str = "") -> str:
            return parsed_data.get(key, default)
        
        def get_bool_value(key: str, default: bool = False) -> bool:
            value = get_value(key).lower()
            return value in ['yes', 'true', '1']
        
        def get_float_value(key: str, default: float = 0.0) -> float:
            try:
                return float(get_value(key, str(default)))
            except (ValueError, TypeError):
                return default
        
        def get_list_value(key: str, default: Optional[List[str]] = None) -> List[str]:
            if default is None:
                default = []
            value = get_value(key)
            if not value or value.lower() in ['none', 'n/a', 'not applicable']:
                return default
            return [item.strip() for item in value.split(',') if item.strip()]
        
        def parse_tool_priorities(priorities_str: str) -> Dict[str, int]:
            """Parse tool priorities from format 'tool1:priority1, tool2:priority2'"""
            priorities = {}
            if not priorities_str or priorities_str.lower() in ['none', 'n/a']:
                return priorities
            
            for item in priorities_str.split(','):
                if ':' in item:
                    tool, priority = item.split(':', 1)
                    try:
                        priorities[tool.strip()] = int(priority.strip())
                    except (ValueError, TypeError):
                        priorities[tool.strip()] = 5  # Default priority
            return priorities
        
        # Build analysis components
        try:
            # Medical Domain Analysis
            medical_analysis = MedicalDomainAnalysis(
                is_medical=get_bool_value('MEDICAL'),
                medical_domain=get_value('MEDICAL_DOMAIN', 'Not Medical'),
                confidence_score=get_float_value('MEDICAL_CONFIDENCE', 0.5),
                medical_entities=get_list_value('MEDICAL_ENTITIES'),
                clinical_context=get_value('CLINICAL_CONTEXT', 'No clinical context identified')
            )
            
            # Complexity Analysis
            complexity_analysis = QueryComplexityAnalysis(
                complexity_level=get_value('COMPLEXITY', 'Moderate'),
                reasoning=get_value('COMPLEXITY_REASONING', 'Standard complexity assessment'),
                estimated_processing_time=get_value('PROCESSING_TIME', 'Standard'),
                requires_multiple_steps=get_bool_value('MULTIPLE_STEPS', True)
            )
            
            # Information Seeking Analysis
            information_analysis = InformationSeekingAnalysis(
                information_type=get_value('INFO_TYPE', 'Factual'),
                specific_needs=get_list_value('SPECIFIC_NEEDS', ['General information']),
                requires_relationships=get_bool_value('RELATIONSHIPS_NEEDED'),
                requires_quantitative_data=get_bool_value('QUANTITATIVE_DATA'),
                temporal_aspect=get_value('TEMPORAL_ASPECT', 'Current')
            )
            
            # Sub-Question Analysis
            sub_questions_list = get_list_value('SUB_QUESTIONS')
            if not sub_questions_list:
                sub_questions_list = [query[:100] + "..." if len(query) > 100 else query]
            
            subquestion_analysis = SubQuestionAnalysis(
                has_multiple_questions=get_bool_value('MULTIPLE_QUESTIONS'),
                sub_questions=sub_questions_list,
                question_dependencies=get_list_value('QUESTION_DEPENDENCIES', ['No dependencies identified']),
                processing_order=get_list_value('PROCESSING_ORDER', ['Process as single question'])
            )
            
            # Tool Requirements Analysis
            recommended_tools = get_list_value('RECOMMENDED_TOOLS', ['vector_search'])
            tool_priorities = parse_tool_priorities(get_value('TOOL_PRIORITIES'))
            
            # Ensure all recommended tools have priorities
            for tool in recommended_tools:
                if tool not in tool_priorities:
                    tool_priorities[tool] = 7  # Default priority
            
            # Build tool reasoning
            tool_reasoning = {}
            primary_reasoning = get_value('PRIMARY_TOOL_REASONING', 'Standard tool selection')
            for i, tool in enumerate(recommended_tools):
                if i == 0:
                    tool_reasoning[tool] = primary_reasoning
                else:
                    tool_reasoning[tool] = f"Supporting tool for comprehensive analysis"
            
            tool_analysis = ToolRequirementAnalysis(
                recommended_tools=recommended_tools,
                tool_priorities=tool_priorities,
                tool_reasoning=tool_reasoning,
                fallback_options=get_list_value('FALLBACK_TOOLS', ['graph_search', 'document_retrieval'])
            )
            
            # Overall Strategy
            overall_strategy = get_value('OVERALL_STRATEGY', 
                                       self._generate_overall_strategy(complexity_analysis, medical_analysis, 
                                                                     information_analysis, subquestion_analysis, tool_analysis))
            
            # Create comprehensive analysis
            comprehensive_analysis = ComprehensiveQueryAnalysis(
                query_id=analysis_id,
                timestamp=datetime.now().isoformat(),
                complexity=complexity_analysis,
                medical_domain=medical_analysis,
                information_seeking=information_analysis,
                sub_questions=subquestion_analysis,
                tool_requirements=tool_analysis,
                overall_strategy=overall_strategy
            )
            
            logger.debug("unified_response_parsed_successfully", 
                        analysis_id=analysis_id,
                        trace_id=trace_id)
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error("unified_response_parsing_failed", 
                        error=str(e),
                        analysis_id=analysis_id,
                        trace_id=trace_id)
            # Re-raise the exception instead of fallback
            raise
    
    def _store_analysis_history(self, analysis_id: str, query: str, 
                               analysis: ComprehensiveQueryAnalysis, 
                               trace_id: Optional[str] = None) -> None:
        """
        Store analysis in history for learning
        
        Args:
            analysis_id: Analysis identifier
            query: The query that was analyzed
            analysis: The comprehensive analysis result
            trace_id: Optional trace ID
        """
        try:
            # Store in history for learning
            self.analysis_history.append({
                "analysis_id": analysis_id,
                "query": query,
                "analysis": analysis,
                "timestamp": datetime.now(),
                "trace_id": trace_id
            })
            
            # Keep only recent history (last 100 analyses)
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
                
            logger.debug("analysis_stored_in_history", 
                        analysis_id=analysis_id,
                        history_size=len(self.analysis_history),
                        trace_id=trace_id)
                        
        except Exception as e:
            logger.error("failed_to_store_analysis_history", 
                        error=str(e),
                        analysis_id=analysis_id,
                        trace_id=trace_id)
    
    def _generate_overall_strategy(self, complexity: QueryComplexityAnalysis, 
                                 medical: MedicalDomainAnalysis,
                                 information: InformationSeekingAnalysis,
                                 subquestions: SubQuestionAnalysis,
                                 tools: ToolRequirementAnalysis) -> str:
        """Generate overall processing strategy"""
        
        strategy_parts = []
        
        # Medical validation first if needed
        if medical.is_medical:
            strategy_parts.append(f"1. Validate medical relevance (confidence: {medical.confidence_score})")
        
        # Handle complexity
        if complexity.complexity_level in ["Complex", "Multi-faceted"]:
            strategy_parts.append("2. Break down into manageable components")
        
        # Handle sub-questions
        if subquestions.has_multiple_questions:
            strategy_parts.append("3. Process sub-questions in dependency order")
        
        # Tool execution strategy
        primary_tools = tools.recommended_tools[:2] if len(tools.recommended_tools) >= 2 else tools.recommended_tools
        strategy_parts.append(f"4. Execute primary tools: {', '.join(primary_tools)}")
        
        # Information synthesis
        if information.information_type in ["Comparative", "Analytical"]:
            strategy_parts.append("5. Synthesize and compare results from multiple sources")
        else:
            strategy_parts.append("5. Consolidate and validate results")
        
        return " -> ".join(strategy_parts)

    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about performed analyses"""
        if not self.analysis_history:
            return {"total_analyses": 0, "common_patterns": []}
        
        total = len(self.analysis_history)
        medical_count = sum(1 for a in self.analysis_history if a["analysis"].medical_domain.is_medical)
        
        complexity_counts = {}
        for analysis in self.analysis_history:
            complexity = analysis["analysis"].complexity.complexity_level
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return {
            "total_analyses": total,
            "medical_queries_percentage": (medical_count / total) * 100 if total > 0 else 0,
            "complexity_distribution": complexity_counts,
            "most_common_complexity": max(complexity_counts.items(), key=lambda x: x[1])[0] if complexity_counts else "Unknown"
        }
