"""
Base Classes and Models for Simple Agentic System

This module contains all the foundational classes needed for the agentic system
without any dependencies on the original agents folder.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
from pydantic.v1 import BaseModel, Field

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from core.logging_config import get_logger

# Initialize logger for base classes
logger = get_logger("base_classes")

# Base workflow state class
class WorkflowState(dict):
    """Simple workflow state dictionary with helper methods"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "timestamp" not in self:
            self["timestamp"] = datetime.now().isoformat()
        logger.debug("workflow_state_initialized", state_keys=list(self.keys()))
    
    def add_result(self, key: str, value: Any):
        """Add a result to the state"""
        self[key] = value
        logger.debug("workflow_state_updated", key=key, value_type=type(value).__name__)
    
    def get_result(self, key: str, default: Any = None):
        """Get a result from the state"""
        result = self.get(key, default)
        logger.debug("workflow_state_accessed", key=key, found=key in self)
        return result

# Agent roles enumeration
class AgentRole(Enum):
    """Simple agent role enumeration"""
    ORCHESTRATOR = "orchestrator"
    VECTOR_RAG = "vector_rag"
    GRAPH_RAG = "graph_rag"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"

# Base agent class
class SecureAgentBase(ABC):
    """Simple secure base class for agents"""
    
    def __init__(self, role: AgentRole):
        self.role = role
        self.agent_id = f"{role.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info("agent_initialized", role=role.value, agent_id=self.agent_id)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        info = {
            "role": self.role.value,
            "agent_id": self.agent_id,
            "status": "active"
        }
        logger.debug("agent_info_requested", agent_id=self.agent_id, role=self.role.value)
        return info

# Pydantic models for structured data
class QueryAnalysis(BaseModel):
    """Simple query analysis model"""
    intent: str = Field(description="Query intent classification")
    complexity: str = Field(description="Query complexity level")
    entity_count: int = Field(description="Number of entities detected")
    has_relationships: bool = Field(description="Whether query involves relationships")

class RoutingDecision(BaseModel):
    """Routing decision with reasoning"""
    route: str = Field(description="Selected route: vector, graph, both, or none")
    confidence: str = Field(description="Confidence level")
    reasoning: str = Field(description="Reasoning for the decision")

class ValidationResult(BaseModel):
    """Validation result model"""
    is_valid: bool = Field(description="Whether content is valid")
    score: float = Field(description="Validation score 0-1")
    feedback: str = Field(description="Validation feedback")

class SynthesisResult(BaseModel):
    """Answer synthesis result"""
    answer: str = Field(description="Synthesized answer")
    sources: List[str] = Field(description="Source documents used")
    confidence: float = Field(description="Confidence in the answer")

# LLM-driven medical validation function
def validate_medical_relevance(query: str, llm) -> Dict[str, Any]:
    """LLM-driven medical relevance validation using secure prompt template"""
    logger.info("medical_validation_started", query_length=len(query))
    
    # Import here to avoid circular imports
    from core.input_sanitization import (
        secure_llm_interaction,
        MEDICAL_VALIDATION_TEMPLATE
    )
    
    # Use secure LLM-driven validation
    logger.info("using_llm_driven_medical_validation")
    
    # Use secure LLM interaction with template and user input
    response = secure_llm_interaction(
        llm=llm,
        template=MEDICAL_VALIDATION_TEMPLATE,
        user_input=query
    )
    
    # Parse LLM response
    response_lower = response.lower().strip()
    
    # Check for medical classification
    is_medical = 'medical' in response_lower and 'non_medical' not in response_lower
    
    logger.debug("llm_medical_validation_result", 
                is_medical=is_medical,
                response_preview=response[:100])
    
    logger.info("query_accepted_medical_llm" if is_medical else "query_rejected_non_medical_llm", 
               query_snippet=query[:50])
    
    return {
        'is_medical': is_medical,
        'quick_response': "I can only help with medical and healthcare-related questions. Please ask about medical conditions, diagnoses, or healthcare topics." if not is_medical else None,
        'validation_method': 'llm_driven',
        'llm_reasoning': response[:200]
    }

# Enhanced LLM-driven query analysis function
def analyze_query_characteristics(query: str, llm=None) -> QueryAnalysis:
    """Enhanced query characteristic analysis using LLM reasoning"""
    logger.info("enhanced_query_analysis_started", query_length=len(query))
    
    # If no LLM provided, fall back to simple analysis
    if llm is None:
        return _simple_query_analysis(query)
    
    # Import here to avoid circular imports
    from core.input_sanitization import secure_llm_interaction
    
    # Enhanced analysis template for intelligent query classification
    analysis_template = """You are a medical database query analyzer. Analyze this query and classify its characteristics:

QUERY: {user_query}

Analyze the query for:

1. INTENT - What is the user trying to accomplish?
   - 'factual': Simple fact lookup or general medical information
   - 'relational': Structured data queries involving patient demographics, findings, relationships
   - 'analytical': Complex analysis, patterns, trends, statistics
   - 'comparison': Comparing conditions, treatments, or demographics

2. COMPLEXITY - How complex is the query?
   - 'simple': Basic questions (under 10 words, single concept)
   - 'complex': Multi-part questions, multiple criteria, aggregations

3. DATABASE INDICATORS - Does this query suggest structured database operations?
   Look for: counts, totals, demographics (age, gender), specific medical findings, exact criteria, filtering
   
4. RELATIONSHIPS - Does the query involve relationships between entities?
   Patient-Finding relationships, demographic-medical correlations, etc.

IMPORTANT: Queries asking for "total number", "count", specific demographics (age=17, male), 
and specific medical findings (effusion) are RELATIONAL queries needing structured database access.

Format your response as:
INTENT: [factual|relational|analytical|comparison]
COMPLEXITY: [simple|complex]  
ENTITY_COUNT: [number of medical entities/concepts]
HAS_RELATIONSHIPS: [true|false]
REASONING: [brief explanation of classification]"""

    try:
        response = secure_llm_interaction(
            llm=llm,
            template=analysis_template,
            user_input=query,
            max_tokens=300,
            temperature=0.1
        )
        
        # Parse LLM response
        intent = 'factual'
        complexity = 'simple'
        entity_count = 1
        has_relationships = False
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip().upper()
            if line.startswith('INTENT:'):
                intent_value = line.split('INTENT:')[1].strip().lower()
                if intent_value in ['factual', 'relational', 'analytical', 'comparison']:
                    intent = intent_value
            elif line.startswith('COMPLEXITY:'):
                complexity_value = line.split('COMPLEXITY:')[1].strip().lower()
                if complexity_value in ['simple', 'complex']:
                    complexity = complexity_value
            elif line.startswith('ENTITY_COUNT:'):
                try:
                    entity_count = int(line.split('ENTITY_COUNT:')[1].strip())
                except:
                    entity_count = 1
            elif line.startswith('HAS_RELATIONSHIPS:'):
                rel_value = line.split('HAS_RELATIONSHIPS:')[1].strip().lower()
                has_relationships = rel_value in ['true', 'yes', '1']
        
        # Additional logic for structured database queries
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in [
            'total number', 'count', 'how many', 'number of',
            'age equals', 'age =', 'gender', 'male', 'female',
            'finding label', 'finding', 'patient', 'diagnosis'
        ]):
            intent = 'relational'
            has_relationships = True
            
        result = QueryAnalysis(
            intent=intent,
            complexity=complexity,
            entity_count=max(entity_count, 1),
            has_relationships=has_relationships
        )
        
        logger.info("llm_query_analysis_completed", 
                   intent=intent, 
                   complexity=complexity, 
                   entity_count=result.entity_count,
                   has_relationships=has_relationships,
                   llm_used=True)
        
        return result
        
    except Exception as e:
        logger.warning("llm_query_analysis_failed", error=str(e))
        return _simple_query_analysis(query)

def _simple_query_analysis(query: str) -> QueryAnalysis:
    """Fallback simple query analysis when LLM is not available"""
    logger.info("simple_query_analysis_started", query_length=len(query))
    
    query_lower = query.lower()
    words = query.split()
    
    # Enhanced intent detection with database query patterns
    if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
        intent = 'comparison'
    elif any(word in query_lower for word in ['relationship', 'correlation', 'association']):
        intent = 'relational'
    elif any(word in query_lower for word in ['analyze', 'pattern', 'trend', 'statistics']):
        intent = 'analytical'
    # Enhanced detection for structured database queries
    elif any(keyword in query_lower for keyword in [
        'total number', 'count', 'how many', 'number of',
        'age equals', 'age =', 'gender', 'male', 'female',
        'finding label', 'finding', 'patient', 'diagnosis'
    ]):
        intent = 'relational'
    else:
        intent = 'factual'
    
    # Determine complexity
    complexity = 'complex' if len(words) > 15 else 'simple'
    
    # Count entities (enhanced heuristic)
    entity_indicators = ['patient', 'disease', 'condition', 'treatment', 'diagnosis', 
                        'finding', 'effusion', 'pneumonia', 'male', 'female', 'age']
    entity_count = sum(1 for indicator in entity_indicators if indicator in query_lower)
    
    # Enhanced relationship detection
    relationship_words = ['relationship', 'correlation', 'association', 'compare', 'versus']
    database_patterns = ['total number', 'count', 'age equals', 'finding label']
    has_relationships = (any(word in query_lower for word in relationship_words) or 
                        any(pattern in query_lower for pattern in database_patterns))
    
    result = QueryAnalysis(
        intent=intent,
        complexity=complexity,
        entity_count=max(entity_count, 1),
        has_relationships=has_relationships
    )
    
    logger.info("simple_query_analysis_completed", 
               intent=intent, 
               complexity=complexity, 
               entity_count=result.entity_count,
               has_relationships=has_relationships,
               llm_used=False)
    
    return result

# Simple document processing functions
def extract_text_from_documents(documents: List[Dict]) -> str:
    """Extract text from document list"""
    logger.debug("text_extraction_started", document_count=len(documents))
    
    if not documents:
        logger.warning("no_documents_provided")
        return ""
    
    texts = []
    for i, doc in enumerate(documents):
        if isinstance(doc, dict):
            # Handle different document formats
            if 'page_content' in doc:
                texts.append(doc['page_content'])
                logger.debug("extracted_from_page_content", doc_index=i)
            elif 'content' in doc:
                texts.append(doc['content'])
                logger.debug("extracted_from_content", doc_index=i)
            elif 'text' in doc:
                texts.append(doc['text'])
                logger.debug("extracted_from_text", doc_index=i)
        elif hasattr(doc, 'page_content'):
            texts.append(doc.page_content)
            logger.debug("extracted_from_object_page_content", doc_index=i)
        else:
            texts.append(str(doc))
            logger.debug("extracted_from_string_conversion", doc_index=i)
    
    result = "\n\n".join(texts)
    logger.info("text_extraction_completed", 
               total_documents=len(documents),
               total_text_length=len(result),
               extracted_chunks=len(texts))
    
    return result

def calculate_simple_quality_score(answer: str, sources: List) -> float:
    """Calculate a simple quality score for an answer"""
    logger.debug("quality_scoring_started", 
                answer_length=len(answer) if answer else 0,
                sources_count=len(sources) if sources else 0)
    
    score = 0.0
    
    # Basic scoring criteria
    if answer and len(answer.strip()) > 50:
        score += 0.4  # Has substantial content
        logger.debug("score_component_added", component="substantial_content", value=0.4)
    
    if sources and len(sources) > 0:
        score += 0.3  # Has sources
        logger.debug("score_component_added", component="has_sources", value=0.3)
    
    if len(answer.split()) > 20:
        score += 0.2  # Adequate length
        logger.debug("score_component_added", component="adequate_length", value=0.2)
    
    # Check for medical terms (indicates relevance)
    medical_terms = ['patient', 'diagnosis', 'treatment', 'medical', 'clinical']
    if any(term in answer.lower() for term in medical_terms):
        score += 0.1
        logger.debug("score_component_added", component="medical_relevance", value=0.1)
    
    final_score = min(score, 1.0)
    logger.info("quality_score_calculated", 
               raw_score=score,
               final_score=final_score,
               answer_words=len(answer.split()) if answer else 0)
    
    return final_score

# Tool registry for simple implementation
class SimpleToolRegistry:
    """Simple tool registry for the agentic system"""
    
    def __init__(self):
        self.tools = {}
        logger.info("tool_registry_initialized")
    
    def register_tool(self, name: str, func: Callable, description: str = ""):
        """Register a tool function"""
        self.tools[name] = {
            'function': func,
            'description': description,
            'registered_at': datetime.now()
        }
        logger.info("tool_registered", name=name, description=description)
    
    def get_tool(self, name: str):
        """Get a registered tool"""
        tool = self.tools.get(name, {}).get('function')
        logger.debug("tool_retrieved", name=name, found=tool is not None)
        return tool
    
    def list_tools(self) -> List[str]:
        """List all registered tools"""
        tools = list(self.tools.keys())
        logger.debug("tools_listed", count=len(tools), tools=tools)
        return tools

# Global tool registry instance
tool_registry = SimpleToolRegistry()

# Register basic tools
tool_registry.register_tool("validate_medical_relevance", validate_medical_relevance, "Validate medical relevance of query")
tool_registry.register_tool("analyze_query_characteristics", analyze_query_characteristics, "Analyze query characteristics")

# Additional result classes for enhanced agentic system
class QueryResult(BaseModel):
    """Query processing result"""
    final_answer: str = Field(description="Final answer to the query")
    sources: List[str] = Field(default_factory=list, description="Source documents used")
    confidence_score: float = Field(default=0.0, description="Confidence in the answer")
    
class AgentResult(BaseModel):
    """Individual agent result"""
    agent_name: str = Field(description="Name of the agent")
    result: str = Field(description="Result from the agent")
    confidence: float = Field(default=0.0, description="Confidence in the result")
    sources: List[str] = Field(default_factory=list, description="Sources used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
