from typing import List, Dict, Any, Optional, cast
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from workflow_state import WorkflowState, ValidationResult
from observability import observability
from logging_config import get_logger
import re

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


# Function calling tools for ValidatorAgent
@tool
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
    if not vector_docs:
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
    
    return RelevanceValidation(
        is_relevant=is_relevant,
        relevance_score=relevance_score,
        reasoning=reasoning,
        key_matches=key_matches[:5]  # Limit to top 5
    )


@tool
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
    if not graph_triples:
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
    
    return RelevanceValidation(
        is_relevant=is_relevant,
        relevance_score=relevance_score,
        reasoning=reasoning,
        key_matches=key_matches[:5]
    )


# Function calling tools for AnswerSynthesisAgent
@tool
def synthesize_answer_from_sources(query: str, vector_docs: List[Dict[str, Any]], 
                                 graph_triples: List[Dict[str, Any]], llm) -> SynthesisResult:
    """
    Synthesize comprehensive answer from vector and graph search results.
    
    Args:
        query: The user query
        vector_docs: List of vector search documents
        graph_triples: List of knowledge graph triples
        llm: LLM instance for synthesis
        
    Returns:
        SynthesisResult with synthesized answer
    """
    if not vector_docs and not graph_triples:
        return SynthesisResult(
            answer="I don't have enough information to answer this query."
        )
    
    # Format vector results
    vector_content = ""
    if vector_docs:
        vector_items = []
        for i, doc in enumerate(vector_docs[:5]):  # Limit to top 5
            content = doc.get("content", "")[:300]  # Limit content length
            vector_items.append(f"Document {i+1}: {content}")
        vector_content = "\n".join(vector_items)
    
    # Format graph results
    graph_content = ""
    if graph_triples:
        graph_items = []
        for i, triple in enumerate(graph_triples[:10]):  # Limit to top 10
            subject = triple.get("subject", "")
            predicate = triple.get("predicate", "")
            obj = triple.get("object", "")
            graph_items.append(f"Relationship {i+1}: {subject} -> {predicate} -> {obj}")
        graph_content = "\n".join(graph_items)
    
    # Create synthesis prompt
    synthesis_prompt = f"""
    Answer this query using the provided information sources.
    
    Query: {query}
    
    Available Information:
    {vector_content if vector_content else "No document information available"}
    
    {graph_content if graph_content else "No relationship information available"}
    
    Instructions:
    1. Provide a comprehensive answer using the available information
    2. Be factual and accurate
    3. If sources are limited, mention this limitation
    
    Answer:"""
    
    try:
        # Get LLM synthesis
        response = llm.invoke(synthesis_prompt)
        answer = response.content.strip()
        
        return SynthesisResult(
            answer=answer if answer else "Unable to synthesize answer from available sources"
        )
        
    except Exception as e:
        logger.warning("synthesis_failed", error=str(e))
        return SynthesisResult(
            answer=f"Error during synthesis: {str(e)}"
        )


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
    
    def validate_results(self, state: WorkflowState) -> WorkflowState:
        """Validate search results using function calling approach"""
        
        with observability.measure_agent_performance("val", cast(Dict[str, Any], state)):
            try:
                query = state["query"]
                vector_docs = state.get("vector_docs", [])
                graph_triples = state.get("graph_triples", [])
                
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
                    "validation_function_calling",
                    passed=overall_passed,
                    confidence=overall_confidence,
                    vector_relevant=vector_validation.is_relevant if vector_docs else None,
                       # uncomment later#######################################
                    #graph_relevant=graph_validation.is_relevant if graph_triples else None,
                       # uncomment later#######################################
                    errors_count=len(errors),
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("validator_function_calling_error", error=str(e), trace_id=state.get('trace_id'))
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


class AnswerSynthesisAgent:
    """
    Function-Calling Answer Synthesis Agent
    Synthesizes comprehensive answers from validated search results using direct tool execution
    Reads: state.vector_docs, state.graph_triples, state.validation_passed
    Writes: state.answer, state.latency_ms["ans"]
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
    
    def synthesize_answer(self, state: WorkflowState) -> WorkflowState:
        """Synthesize final answer using function calling approach"""
        
        with observability.measure_agent_performance("ans", cast(Dict[str, Any], state)):
            try:
                query = state["query"]
                vector_docs = state.get("vector_docs", [])
                graph_triples = state.get("graph_triples", [])
                validation_passed = state.get("validation_passed", True)
                
                # Only synthesize if validation passed
                if not validation_passed:
                    state["answer"] = "Unable to provide a reliable answer due to validation concerns with the retrieved information."
                    state["status"] = "completed_with_validation_issues"
                    return state
                
                # Check if we have any data to synthesize
                if not vector_docs and not graph_triples:
                    state["answer"] = "I don't have enough information to answer this query based on the available data sources."
                    state["status"] = "completed_no_data"
                    return state
                
                # Synthesize answer from sources using tool
                synthesis_result = synthesize_answer_from_sources.invoke({
                    "query": query,
                    "vector_docs": vector_docs or [],
                    "graph_triples": graph_triples or [],
                    "llm": self.llm
                })
                
                # Update state with synthesis results
                state["answer"] = synthesis_result.answer
                state["status"] = "completed"
                
                logger.info(
                    "answer_synthesis_function_calling",
                    answer_length=len(synthesis_result.answer),
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("synthesis_function_calling_error", error=str(e), trace_id=state.get('trace_id'))
                state["answer"] = f"I encountered an error while synthesizing the answer: {str(e)}"
                state["status"] = "failed"
                errors = state.get("errors") or []
                state["errors"] = errors + [f"Synthesis error: {str(e)}"]
                return state
