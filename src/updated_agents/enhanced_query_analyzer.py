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
from langchain_core.prompts import PromptTemplate

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
    confidence_assessment: float = Field(description="Overall confidence in analysis (0-1)")

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
    def analyze_query_comprehensive(self, query: str) -> ComprehensiveQueryAnalysis:
        """
        Perform comprehensive analysis of the user query
        
        Args:
            query: The user query to analyze
            
        Returns:
            ComprehensiveQueryAnalysis: Complete structured analysis
        """
        logger.info("comprehensive_query_analysis_started", query_length=len(query))
        
        # Security check - sanitize input and detect injection
        if detect_prompt_injection(query):
            logger.warning("prompt_injection_detected_in_query")
            raise ValueError("Potential prompt injection detected in query")
        
        sanitized_query = sanitize_user_input(query)
        
        # Generate unique analysis ID
        analysis_id = f"analysis_{self.analysis_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.analysis_count += 1
        
        try:
            # Perform different aspects of analysis
            complexity_analysis = self._analyze_complexity(sanitized_query)
            medical_analysis = self._analyze_medical_domain(sanitized_query)
            information_analysis = self._analyze_information_seeking(sanitized_query)
            subquestion_analysis = self._analyze_sub_questions(sanitized_query)
            tool_analysis = self._analyze_tool_requirements(sanitized_query, medical_analysis, complexity_analysis)
            
            # Generate overall strategy
            overall_strategy = self._generate_overall_strategy(
                complexity_analysis, medical_analysis, information_analysis, 
                subquestion_analysis, tool_analysis
            )
            
            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(
                complexity_analysis, medical_analysis, information_analysis
            )
            
            # Create comprehensive analysis
            comprehensive_analysis = ComprehensiveQueryAnalysis(
                query_id=analysis_id,
                timestamp=datetime.now().isoformat(),
                complexity=complexity_analysis,
                medical_domain=medical_analysis,
                information_seeking=information_analysis,
                sub_questions=subquestion_analysis,
                tool_requirements=tool_analysis,
                overall_strategy=overall_strategy,
                confidence_assessment=confidence
            )
            
            # Store in history for learning
            self.analysis_history.append({
                "analysis_id": analysis_id,
                "query": sanitized_query,
                "analysis": comprehensive_analysis,
                "timestamp": datetime.now()
            })
            
            # Keep only recent history (last 100 analyses)
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
            
            logger.info("comprehensive_query_analysis_completed", 
                       analysis_id=analysis_id,
                       overall_strategy=overall_strategy,
                       confidence=confidence)
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error("comprehensive_query_analysis_failed", error=str(e))
            raise
    
    def _analyze_complexity(self, query: str) -> QueryComplexityAnalysis:
        """Analyze query complexity using LLM"""
        logger.debug("analyzing_query_complexity")
        
        complexity_template = """
        Analyze the complexity of this query and provide structured assessment.
        
        Consider these factors:
        - Number of concepts involved
        - Depth of analysis required
        - Multiple data sources needed
        - Relationship complexity
        - Processing steps required
        
        Query to analyze: {user_query}
        
        Provide assessment in this format:
        Complexity Level: [Simple/Moderate/Complex/Multi-faceted]
        Reasoning: [Detailed explanation]
        Estimated Processing Time: [Quick/Standard/Extended/Comprehensive]
        Requires Multiple Steps: [Yes/No]
        """
        
        try:
            response = secure_llm_interaction(
                self.llm, 
                complexity_template, 
                query
            )
            
            # Parse response and create structured output
            return self._parse_complexity_response(response)
            
        except Exception as e:
            logger.error("complexity_analysis_failed", error=str(e))
            # Return default complexity analysis
            return QueryComplexityAnalysis(
                complexity_level="Moderate",
                reasoning="Analysis failed, using default assessment",
                estimated_processing_time="Standard",
                requires_multiple_steps=True
            )
    
    def _analyze_medical_domain(self, query: str) -> MedicalDomainAnalysis:
        """Analyze medical domain and relevance"""
        logger.debug("analyzing_medical_domain")
        
        medical_template = """
        Analyze this query for medical/healthcare relevance and domain classification.
        
        Determine:
        1. Is this query medical/healthcare related?
        2. What specific medical domain (radiology, cardiology, oncology, etc.)?
        3. What medical entities are present?
        4. What is the clinical context?
        
        Query to analyze: {user_query}
        
        Provide assessment in this format:
        Is Medical: [Yes/No]
        Medical Domain: [Specific domain or "Not Medical"]
        Confidence Score: [0.0-1.0]
        Medical Entities: [List of medical terms found]
        Clinical Context: [Description of clinical scenario]
        """
        
        try:
            response = secure_llm_interaction(
                self.llm,
                medical_template,
                query
            )
            
            return self._parse_medical_response(response)
            
        except Exception as e:
            logger.error("medical_domain_analysis_failed", error=str(e))
            return MedicalDomainAnalysis(
                is_medical=False,
                medical_domain="Unknown",
                confidence_score=0.0,
                medical_entities=[],
                clinical_context="Analysis failed"
            )
    
    def _analyze_information_seeking(self, query: str) -> InformationSeekingAnalysis:
        """Analyze what type of information is being sought"""
        logger.debug("analyzing_information_seeking")
        
        info_template = """
        Analyze what type of information this query is seeking.
        
        Consider:
        - Type of information (Factual, Comparative, Analytical, Procedural, Diagnostic)
        - Specific information needs
        - Whether relationships between entities are needed
        - Whether quantitative/statistical data is required
        - Time-related aspects
        
        Query to analyze: {user_query}
        
        Provide assessment in this format:
        Information Type: [Factual/Comparative/Analytical/Procedural/Diagnostic]
        Specific Needs: [List of specific information requirements]
        Requires Relationships: [Yes/No]
        Requires Quantitative Data: [Yes/No]
        Temporal Aspect: [Current/Historical/Trending/Predictive/Not Applicable]
        """
        
        try:
            response = secure_llm_interaction(
                self.llm,
                info_template,
                query
            )
            
            return self._parse_information_response(response)
            
        except Exception as e:
            logger.error("information_seeking_analysis_failed", error=str(e))
            return InformationSeekingAnalysis(
                information_type="Factual",
                specific_needs=["General information"],
                requires_relationships=False,
                requires_quantitative_data=False,
                temporal_aspect="Current"
            )
    
    def _analyze_sub_questions(self, query: str) -> SubQuestionAnalysis:
        """Analyze sub-questions within the main query"""
        logger.debug("analyzing_sub_questions")
        
        subq_template = """
        Analyze this query for multiple sub-questions and their relationships.
        
        Identify:
        - Whether the query contains multiple distinct questions
        - Individual sub-questions that need separate answers
        - Dependencies between sub-questions
        - Optimal processing order
        
        Query to analyze: {user_query}
        
        Provide assessment in this format:
        Has Multiple Questions: [Yes/No]
        Sub-questions: [List each sub-question separately]
        Question Dependencies: [Describe how questions relate]
        Processing Order: [Recommended order for addressing questions]
        """
        
        try:
            response = secure_llm_interaction(
                self.llm,
                subq_template,
                query
            )
            
            return self._parse_subquestion_response(response)
            
        except Exception as e:
            logger.error("sub_question_analysis_failed", error=str(e))
            return SubQuestionAnalysis(
                has_multiple_questions=False,
                sub_questions=[query],
                question_dependencies=["No dependencies"],
                processing_order=["Process as single question"]
            )
    
    def _analyze_tool_requirements(self, query: str, medical_analysis: MedicalDomainAnalysis, 
                                 complexity_analysis: QueryComplexityAnalysis) -> ToolRequirementAnalysis:
        """Analyze what tools would be most effective"""
        logger.debug("analyzing_tool_requirements")
        
        # Available tools in the system
        available_tools = [
            "vector_search", "graph_search", "both_searches",
            "medical_validation", "query_analysis", "document_retrieval",
            "relationship_analysis", "statistical_analysis"
        ]
        
        tool_template = """
        Based on this query analysis, recommend the most effective tools and processing approach.
        
        Query: {user_query}
        Medical Domain: {medical_domain}
        Complexity: {complexity_level}
        Is Medical: {is_medical}
        
        Available tools: {available_tools}
        
        Provide recommendations in this format:
        Recommended Tools: [List of tools in order of importance]
        Tool Priorities: [Rate each tool 1-10 for importance]
        Tool Reasoning: [Explain why each tool is recommended]
        Fallback Options: [Alternative tools if primary options fail]
        """
        
        try:
            response = secure_llm_interaction(
                self.llm,
                tool_template,
                query,
                medical_domain=medical_analysis.medical_domain,
                complexity_level=complexity_analysis.complexity_level,
                is_medical=medical_analysis.is_medical,
                available_tools=", ".join(available_tools)
            )
            
            return self._parse_tool_response(response)
            
        except Exception as e:
            logger.error("tool_requirements_analysis_failed", error=str(e))
            return ToolRequirementAnalysis(
                recommended_tools=["vector_search", "medical_validation"],
                tool_priorities={"vector_search": 8, "medical_validation": 9},
                tool_reasoning={"vector_search": "Default search", "medical_validation": "Security check"},
                fallback_options=["graph_search"]
            )
    
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
    
    def _calculate_overall_confidence(self, complexity: QueryComplexityAnalysis,
                                    medical: MedicalDomainAnalysis,
                                    information: InformationSeekingAnalysis) -> float:
        """Calculate overall confidence in the analysis"""
        
        confidence_factors = []
        
        # Medical classification confidence
        confidence_factors.append(medical.confidence_score)
        
        # Complexity assessment confidence
        if complexity.complexity_level in ["Simple", "Moderate"]:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.7)
        
        # Information type clarity
        if information.information_type in ["Factual", "Procedural"]:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _parse_complexity_response(self, response: str) -> QueryComplexityAnalysis:
        """Parse LLM response for complexity analysis"""
        # Simple parsing - in production, would use more robust parsing
        lines = response.strip().split('\n')
        
        complexity_level = "Moderate"
        reasoning = "Analysis completed"
        processing_time = "Standard"
        multiple_steps = True
        
        for line in lines:
            if "complexity level:" in line.lower():
                complexity_level = line.split(':', 1)[1].strip()
            elif "reasoning:" in line.lower():
                reasoning = line.split(':', 1)[1].strip()
            elif "processing time:" in line.lower():
                processing_time = line.split(':', 1)[1].strip()
            elif "multiple steps:" in line.lower():
                multiple_steps = "yes" in line.lower()
        
        return QueryComplexityAnalysis(
            complexity_level=complexity_level,
            reasoning=reasoning,
            estimated_processing_time=processing_time,
            requires_multiple_steps=multiple_steps
        )
    
    def _parse_medical_response(self, response: str) -> MedicalDomainAnalysis:
        """Parse LLM response for medical analysis"""
        lines = response.strip().split('\n')
        
        is_medical = False
        medical_domain = "Not Medical"
        confidence_score = 0.0
        medical_entities = []
        clinical_context = "No clinical context"
        
        for line in lines:
            if "is medical:" in line.lower():
                is_medical = "yes" in line.lower()
            elif "medical domain:" in line.lower():
                medical_domain = line.split(':', 1)[1].strip()
            elif "confidence score:" in line.lower():
                try:
                    confidence_score = float(line.split(':', 1)[1].strip())
                except:
                    confidence_score = 0.5
            elif "medical entities:" in line.lower():
                entities_str = line.split(':', 1)[1].strip()
                medical_entities = [e.strip() for e in entities_str.split(',') if e.strip()]
            elif "clinical context:" in line.lower():
                clinical_context = line.split(':', 1)[1].strip()
        
        return MedicalDomainAnalysis(
            is_medical=is_medical,
            medical_domain=medical_domain,
            confidence_score=confidence_score,
            medical_entities=medical_entities,
            clinical_context=clinical_context
        )
    
    def _parse_information_response(self, response: str) -> InformationSeekingAnalysis:
        """Parse LLM response for information seeking analysis"""
        lines = response.strip().split('\n')
        
        information_type = "Factual"
        specific_needs = ["General information"]
        requires_relationships = False
        requires_quantitative = False
        temporal_aspect = "Current"
        
        for line in lines:
            if "information type:" in line.lower():
                information_type = line.split(':', 1)[1].strip()
            elif "specific needs:" in line.lower():
                needs_str = line.split(':', 1)[1].strip()
                specific_needs = [n.strip() for n in needs_str.split(',') if n.strip()]
            elif "requires relationships:" in line.lower():
                requires_relationships = "yes" in line.lower()
            elif "requires quantitative:" in line.lower():
                requires_quantitative = "yes" in line.lower()
            elif "temporal aspect:" in line.lower():
                temporal_aspect = line.split(':', 1)[1].strip()
        
        return InformationSeekingAnalysis(
            information_type=information_type,
            specific_needs=specific_needs,
            requires_relationships=requires_relationships,
            requires_quantitative_data=requires_quantitative,
            temporal_aspect=temporal_aspect
        )
    
    def _parse_subquestion_response(self, response: str) -> SubQuestionAnalysis:
        """Parse LLM response for sub-question analysis"""
        lines = response.strip().split('\n')
        
        has_multiple = False
        sub_questions = []
        dependencies = ["No dependencies"]
        processing_order = ["Process as single question"]
        
        for line in lines:
            if "has multiple questions:" in line.lower():
                has_multiple = "yes" in line.lower()
            elif "sub-questions:" in line.lower():
                questions_str = line.split(':', 1)[1].strip()
                sub_questions = [q.strip() for q in questions_str.split(',') if q.strip()]
            elif "question dependencies:" in line.lower():
                deps_str = line.split(':', 1)[1].strip()
                dependencies = [d.strip() for d in deps_str.split(',') if d.strip()]
            elif "processing order:" in line.lower():
                order_str = line.split(':', 1)[1].strip()
                processing_order = [o.strip() for o in order_str.split(',') if o.strip()]
        
        return SubQuestionAnalysis(
            has_multiple_questions=has_multiple,
            sub_questions=sub_questions if sub_questions else [response[:100]],
            question_dependencies=dependencies,
            processing_order=processing_order
        )
    
    def _parse_tool_response(self, response: str) -> ToolRequirementAnalysis:
        """Parse LLM response for tool requirements"""
        lines = response.strip().split('\n')
        
        recommended_tools = ["vector_search"]
        tool_priorities = {"vector_search": 8}
        tool_reasoning = {"vector_search": "Default search tool"}
        fallback_options = ["graph_search"]
        
        for line in lines:
            if "recommended tools:" in line.lower():
                tools_str = line.split(':', 1)[1].strip()
                recommended_tools = [t.strip() for t in tools_str.split(',') if t.strip()]
            elif "fallback options:" in line.lower():
                fallback_str = line.split(':', 1)[1].strip()
                fallback_options = [f.strip() for f in fallback_str.split(',') if f.strip()]
        
        # Set default priorities and reasoning for recommended tools
        for tool in recommended_tools:
            if tool not in tool_priorities:
                tool_priorities[tool] = 7
            if tool not in tool_reasoning:
                tool_reasoning[tool] = f"Recommended for this query type"
        
        return ToolRequirementAnalysis(
            recommended_tools=recommended_tools,
            tool_priorities=tool_priorities,
            tool_reasoning=tool_reasoning,
            fallback_options=fallback_options
        )
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get analysis history for learning purposes"""
        return self.analysis_history
    
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
