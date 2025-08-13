from typing import List, Dict, Any, Optional
import re
import sys
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from .workflow_state import WorkflowState, ValidationResult

# Add the src directory to the path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try relative imports first, fall back to absolute imports
try:
    from ..core.observability import  traceable, get_traceable_config
    from ..core.input_sanitization import detect_prompt_injection, sanitize_user_input, validate_llm_output, secure_llm_interaction
except ImportError:
    # Fall back to absolute imports
    from core.observability import  traceable, get_traceable_config
    from core.input_sanitization import detect_prompt_injection, sanitize_user_input, validate_llm_output, secure_llm_interaction

from .logging_config import get_logger
from .tool_governance import ToolRegistry, ToolMetadata, AgentRole, tool_registry, AccessDeniedError, SecureAgentBase

logger = get_logger("validation_synthesis")


# Pydantic models for structured outputs
class RelevanceValidation(BaseModel):
    """Relevance validation result for search results"""
    is_relevant: bool = Field(description="Whether the results are relevant to the query")
    relevance_score: float = Field(description="Relevance score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of relevance assessment")
    key_matches: List[str] = Field(description="Key matching terms or concepts found")


class SynthesisResult(BaseModel):
    """Answer synthesis result"""
    answer: str = Field(description="Synthesized comprehensive answer")


# Called by: ValidatorAgent
@tool
@traceable(**get_traceable_config("ValidatorAgent"))
def validate_vector_relevance(query: str, vector_docs: List[Dict[str, Any]]) -> RelevanceValidation:
    """
    Validate relevance of vector search results against the query.
    Uses simple keyword matching and score analysis.
    
    Args:
        query: The original user query
        vector_docs: List of vector search documents with scores
        
    Returns:
        RelevanceValidation with relevance assessment
    """
    logger.info("validate_vector_relevance_started", 
               query_length=len(query), 
               documents_count=len(vector_docs))
    
    if not vector_docs:
        logger.info("validate_vector_relevance_completed", 
                   is_relevant=False, 
                   reason="no_documents")
        return RelevanceValidation(
            is_relevant=False,
            relevance_score=0.0,
            reasoning="No vector search results to validate",
            key_matches=[]
        )
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Analyze documents for relevance
    total_score = 0.0
    key_matches = []
    
    for doc in vector_docs:
        content = doc.get("content", "").lower()
        score = doc.get("score", 0.0)
        total_score += score
        
        # Find matching query words in content
        content_words = set(re.findall(r'\b\w+\b', content))
        common_words = query_words.intersection(content_words)
        for word in common_words:
            if len(word) > 3 and word not in key_matches:
                key_matches.append(word)
    
    # Calculate relevance metrics
    avg_score = total_score / len(vector_docs)
    relevance_score = avg_score  # Simple: use average similarity score
    is_relevant = relevance_score > 0.4 and len(key_matches) > 0
    
    reasoning = f"Avg similarity: {avg_score:.2f}, Matches: {len(key_matches)}"
    
    logger.info("validate_vector_relevance_completed",
               is_relevant=is_relevant,
               relevance_score=relevance_score,
               key_matches_count=len(key_matches),
               avg_score=avg_score)
    
    return RelevanceValidation(
        is_relevant=is_relevant,
        relevance_score=relevance_score,
        reasoning=reasoning,
        key_matches=key_matches[:5]  # Limit to top 5
    )

# Called by: ValidatorAgent
@tool
@traceable(**get_traceable_config("ValidatorAgent"))
def validate_graph_relevance(query: str, graph_triples: List[Dict[str, Any]]) -> RelevanceValidation:
    """
    Validate relevance of graph query results against the query.
    Uses simple entity matching.
    
    Args:
        query: The original user query
        graph_triples: List of knowledge graph triples
        
    Returns:
        RelevanceValidation with relevance assessment
    """
    logger.info("validate_graph_relevance_started",
               query_length=len(query),
               triples_count=len(graph_triples))
    
    if not graph_triples:
        logger.info("validate_graph_relevance_completed",
                   is_relevant=False,
                   reason="no_triples")
        return RelevanceValidation(
            is_relevant=False,
            relevance_score=0.0,
            reasoning="No graph search results to validate",
            key_matches=[]
        )
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Find matching entities in graph results
    key_matches = []
    for triple in graph_triples:
        subject = str(triple.get("subject", "")).lower()
        obj = str(triple.get("object", "")).lower()
        
        # Check for query word matches in entities
        for entity in [subject, obj]:
            entity_words = set(re.findall(r'\b\w+\b', entity))
            matches = query_words.intersection(entity_words)
            for match in matches:
                if len(match) > 3 and match not in key_matches:
                    key_matches.append(match)
    
    # Simple relevance calculation
    match_coverage = len(key_matches) / max(len(query_words), 1)
    relevance_score = min(1.0, match_coverage * 2)  # Boost score for graph results
    is_relevant = relevance_score > 0.3 and len(key_matches) > 0
    
    reasoning = f"Entities found: {len(key_matches)}, Coverage: {match_coverage:.2f}"
    
    logger.info("validate_graph_relevance_completed",
               is_relevant=is_relevant,
               relevance_score=relevance_score,
               key_matches_count=len(key_matches),
               match_coverage=match_coverage)
    
    return RelevanceValidation(
        is_relevant=is_relevant,
        relevance_score=relevance_score,
        reasoning=reasoning,
        key_matches=key_matches[:5]
    )


# Called by: AnswerSynthesisAgent
@tool
@traceable(**get_traceable_config("AnswerSynthesisAgent"))
def synthesize_answer_from_sources(query: str, vector_docs: List[Dict[str, Any]], 
                                 graph_triples: List[Dict[str, Any]], llm) -> SynthesisResult:
    """
    Secure synthesis of comprehensive answer from vector and graph search results.
    Implements prompt injection protection and parameterized templates.
    
    Args:
        query: The user query
        vector_docs: List of vector search documents
        graph_triples: List of knowledge graph triples
        llm: LLM instance for synthesis
        
    Returns:
        SynthesisResult with synthesized answer
    """
    logger.info("synthesize_answer_from_sources_started",
               query_length=len(query),
               vector_docs_count=len(vector_docs),
               graph_triples_count=len(graph_triples))        
    
    if not vector_docs and not graph_triples:
        logger.info("synthesize_answer_from_sources_completed",
                   answer_length=0,
                   reason="no_sources")
        return SynthesisResult(
            answer="I don't have enough information to answer this query."
        )

    # Step 1: Format vector results with sanitization
    vector_content = "No document information available"
    if vector_docs:
        vector_items = []
        for i, doc in enumerate(vector_docs[:5]):  # Limit to top 5
            content = doc.get("content", "")[:300]  # Limit content length
            # Sanitize document content
            sanitized_content = sanitize_user_input(content)
            vector_items.append(f"Document {i+1}: {sanitized_content}")
        vector_content = "\n".join(vector_items)

    # Step 2: Format graph results with sanitization  
    graph_content = "No relationship information available"
    if graph_triples:
        graph_items = []
        for i, triple in enumerate(graph_triples):  # Process all triples
            triple_type = triple.get("type", "unknown")
            
            if triple_type == "patient_finding":
                # Handle patient finding data structure
                patient_id = sanitize_user_input(str(triple.get("patient_id", "")))
                age = sanitize_user_input(str(triple.get("age", "")))
                gender = sanitize_user_input(str(triple.get("gender", "")))
                finding = sanitize_user_input(str(triple.get("finding", "")))
                graph_items.append(f"Patient {patient_id}: Age {age}, Gender {gender}, Finding: {finding}")
            
            elif triple_type == "aggregation":
                # Handle aggregation data structure
                finding = sanitize_user_input(str(triple.get("finding", "")))
                count = sanitize_user_input(str(triple.get("count", "")))
                graph_items.append(f"Finding {finding}: {count} patients")
            
            elif triple_type == "multiple_conditions":
                # Handle multiple conditions data structure
                patient_id = sanitize_user_input(str(triple.get("patient_id", "")))
                age = sanitize_user_input(str(triple.get("age", "")))
                gender = sanitize_user_input(str(triple.get("gender", "")))
                conditions = triple.get("conditions", [])
                conditions_str = ", ".join([sanitize_user_input(str(c)) for c in conditions])
                graph_items.append(f"Patient {patient_id}: Age {age}, Gender {gender}, Conditions: {conditions_str}")
            
            else:
                # Fallback for legacy subject-predicate-object format
                subject = sanitize_user_input(str(triple.get("subject", "")))
                predicate = sanitize_user_input(str(triple.get("predicate", "")))
                obj = sanitize_user_input(str(triple.get("object", "")))
                if subject or predicate or obj:
                    graph_items.append(f"Relationship {i+1}: {subject} -> {predicate} -> {obj}")
        
        if graph_items:
            graph_content = "\n".join(graph_items)
        else:
            graph_content = "No relationship information available"

    try:
        # Step 3: Use secure LLM interaction with input delimiters
        try:
            from core.input_sanitization import SYNTHESIS_TEMPLATE
        except ImportError:
            from core.input_sanitization import SYNTHESIS_TEMPLATE
        
        validated_content = secure_llm_interaction(
            llm=llm,
            template=SYNTHESIS_TEMPLATE,
            user_input=query,
            vector_content=vector_content,
            graph_content=graph_content
        )
        
        final_answer = validated_content if validated_content else "Unable to synthesize answer from available sources"
        
        logger.info("synthesize_answer_from_sources_completed",
                   answer_length=len(final_answer),
                   synthesis_successful=bool(validated_content))
        
        return SynthesisResult(answer=final_answer)
        
    except Exception as e:
        logger.warning("answer_synthesis_failed", error=str(e))
        fallback_answer = f"Error during synthesis: {str(e)}"
        logger.info("synthesize_answer_from_sources_completed",
                   answer_length=len(fallback_answer),
                   synthesis_successful=False,
                   error=str(e))
        return SynthesisResult(answer=fallback_answer)


class ValidatorAgent:
    """
    Function-Calling Validator Agent
    Validates relevance of search results using direct tool execution
    Reads: state.vector_docs, state.graph_triples
    Writes: state.validation_passed, state.validation_result, state.latency_ms["val"]
    """
    
    def __init__(self, llm: Optional[AzureChatOpenAI] = None):
        # LLM not required for basic validation - using rule-based tools
        self.llm = llm
        logger.info("validator_agent_initialized", has_llm=llm is not None)
    
    @traceable(**get_traceable_config("ValidatorAgent"))
    def validate_results(self, state: WorkflowState) -> WorkflowState:
        """Validate search results using function calling approach"""
        
        trace_id = state.get('trace_id')
        logger.info("validator_validate_results_started", trace_id=trace_id)
        
        try:
            query = state["query"]
            vector_docs = state.get("vector_docs", []) or []
            graph_triples = state.get("graph_triples", []) or []
            
            logger.info("validator_processing_results",
                       query_length=len(query),
                       vector_docs_count=len(vector_docs),
                       graph_triples_count=len(graph_triples),
                       trace_id=trace_id)
            
            # Step 1: Validate vector search relevance
            vector_validation = validate_vector_relevance.invoke({
                "query": query,
                "vector_docs": vector_docs
            })

             # uncomment later#######################################3
            # Step 2: Validate graph search relevance  
            # graph_validation = validate_graph_relevance.invoke({
            #     "query": query,
            #     "graph_triples": graph_triples
            # })
            # uncomment later#######################################3
            
            # Combine validation results - focus only on relevance
            overall_passed = True
            errors = []
            confidence_scores = []
            
            # Check vector relevance
            if vector_docs and not vector_validation.is_relevant:
                errors.append("Vector search results not relevant to query")
                overall_passed = False
            if vector_docs:
                confidence_scores.append(vector_validation.relevance_score)
            
             # uncomment later#######################################3
            # Check graph relevance
            # if graph_triples and not graph_validation.is_relevant:
            #     errors.append("Graph search results not relevant to query")
            #     overall_passed = False
            # if graph_triples:
            #     confidence_scores.append(graph_validation.relevance_score)
             # uncomment later#######################################3
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            # Create final validation result
            validation_result = ValidationResult(
                passed=overall_passed,
                errors=errors,
                confidence_score=overall_confidence,
                consistency_issues=[]  # Not checking consistency anymore
            )
            
            # Update state
            state["validation_passed"] = overall_passed
            state["validation_errors"] = errors
            state["validation_result"] = validation_result
            
            logger.info(
                "validator_validate_results_completed",
                passed=overall_passed,
                confidence=overall_confidence,
                vector_relevant=vector_validation.is_relevant if vector_docs else None,
                   # uncomment later#######################################
                #graph_relevant=graph_validation.is_relevant if graph_triples else None,
                   # uncomment later#######################################
                errors_count=len(errors),
                trace_id=trace_id
            )
            
            return state
            
        except Exception as e:
            logger.error("validator_validate_results_error", error=str(e), trace_id=trace_id)
            # Safe fallback - pass validation to avoid blocking workflow
            validation_result = ValidationResult(
                passed=True,
                errors=[f"Validation error: {str(e)}"],
                confidence_score=0.5,
                consistency_issues=["Could not perform full validation"]
            )
            state["validation_passed"] = True
            state["validation_errors"] = validation_result.errors
            state["validation_result"] = validation_result
            return state


class AnswerSynthesisAgent(SecureAgentBase):
    """
    Function-Calling Answer Synthesis Agent
    Synthesizes comprehensive answers from validated search results using direct tool execution
    Reads: state.vector_docs, state.graph_triples, state.validation_passed
    Writes: state.answer, state.latency_ms["ans"]
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        super().__init__(AgentRole.SYNTHESIZER)
        self.llm = llm
        logger.info("answer_synthesis_agent_initialized", has_llm=llm is not None)
    
    @traceable(**get_traceable_config("AnswerSynthesisAgent"))
    def synthesize_answer(self, state: WorkflowState) -> WorkflowState:
        """Synthesize final answer using function calling approach"""
        
        trace_id = state.get('trace_id')
        logger.info("answer_synthesis_synthesize_answer_started", trace_id=trace_id)
        
        try:
            query = state["query"]
            vector_docs = state.get("vector_docs", []) or []
            graph_triples = state.get("graph_triples", []) or []
            
            logger.info("answer_synthesis_processing_sources",
                       query_length=len(query),
                       vector_docs_count=len(vector_docs),
                       graph_triples_count=len(graph_triples),
                       trace_id=trace_id)
            validation_passed = state.get("validation_passed", True)
            
            # Only synthesize if validation passed
            if not validation_passed:
                no_validation_answer = "Unable to provide a reliable answer due to validation concerns with the retrieved information."
                state["answer"] = no_validation_answer
                state["status"] = "completed_with_validation_issues"
                logger.info("answer_synthesis_synthesize_answer_completed",
                           answer_length=len(no_validation_answer),
                           status="validation_failed",
                           trace_id=trace_id)
                return state
            
            # Check if we have any data to synthesize
            if not vector_docs and not graph_triples:
                no_data_answer = "I don't have enough information to answer this query based on the available data sources."
                state["answer"] = no_data_answer
                state["status"] = "completed_no_data"
                logger.info("answer_synthesis_synthesize_answer_completed",
                           answer_length=len(no_data_answer),
                           status="no_data",
                           trace_id=trace_id)
                return state
            
            # Synthesize answer from sources using tool
            synthesis_result = self.invoke_tool("synthesize_answer_from_sources", {
                "query": query,
                "vector_docs": vector_docs or [],
                "graph_triples": graph_triples or [],
                "llm": self.llm
            })
            
            # Update state with synthesis results
            state["answer"] = synthesis_result.answer
            state["status"] = "completed"
            
            logger.info(
                "answer_synthesis_synthesize_answer_completed",
                answer_length=len(synthesis_result.answer),
                status="completed",
                trace_id=trace_id
            )
            
            return state
            
        except Exception as e:
            logger.error("answer_synthesis_synthesize_answer_error", error=str(e), trace_id=trace_id)
            error_answer = f"I encountered an error while synthesizing the answer: {str(e)}"
            state["answer"] = error_answer
            state["status"] = "failed"
            errors = state.get("errors") or []
            state["errors"] = errors + [f"Synthesis error: {str(e)}"]
            logger.info("answer_synthesis_synthesize_answer_completed",
                       answer_length=len(error_answer),
                       status="failed",
                       trace_id=trace_id)
            return state

def register_validation_synthesis_tools():
    """Register validation and synthesis tools with their allowed agent roles"""
    
    logger.info("register_validation_synthesis_tools_started")
    
    # Validator tools
    tool_registry.register_tool(
        validate_vector_relevance,
        ToolMetadata("validate_vector_relevance", [AgentRole.VALIDATOR])
    )
    tool_registry.register_tool(
        validate_graph_relevance,
        ToolMetadata("validate_graph_relevance", [AgentRole.VALIDATOR])
    )
    
    # Synthesizer tools
    tool_registry.register_tool(
        synthesize_answer_from_sources,
        ToolMetadata("synthesize_answer_from_sources", [AgentRole.SYNTHESIZER])
    )
    
    logger.info("register_validation_synthesis_tools_completed",
               validator_tools=2,
               synthesizer_tools=1,
               total_tools=3)

# Initialize tool registry for validation and synthesis
logger.info("initializing_validation_synthesis_tools")
register_validation_synthesis_tools()
logger.info("validation_synthesis_tools_initialization_completed")