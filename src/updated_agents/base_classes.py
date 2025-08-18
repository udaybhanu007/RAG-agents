"""
Base Classes and Models for Simple Agentic System

This module contains all the foundational classes needed for the agentic system
without any dependencies on the original agents folder.
"""

import os
import sys
from typing import Dict, Any, List, Optional
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

# Simple medical validation function
def validate_medical_relevance(query: str) -> Dict[str, Any]:
    """Simple medical relevance validation"""
    logger.info("medical_validation_started", query_length=len(query))
    
    medical_keywords = [
        'medical', 'health', 'disease', 'diagnosis', 'treatment', 'symptom',
        'patient', 'doctor', 'hospital', 'medicine', 'drug', 'therapy',
        'pneumonia', 'covid', 'xray', 'chest', 'lung', 'heart', 'cancer'
    ]
    
    query_lower = query.lower()
    is_medical = any(keyword in query_lower for keyword in medical_keywords)
    
    logger.debug("medical_keywords_check", 
                keywords_found=[kw for kw in medical_keywords if kw in query_lower],
                is_medical=is_medical)
    
    if not is_medical:
        logger.info("query_rejected_non_medical", query_snippet=query[:50])
        return {
            'is_medical': False,
            'quick_response': "I can only help with medical and healthcare-related questions. Please ask about medical conditions, diagnoses, or healthcare topics."
        }
    
    logger.info("query_accepted_medical", query_snippet=query[:50])
    return {
        'is_medical': True,
        'quick_response': None
    }

# Simple query analysis function
def analyze_query_characteristics(query: str) -> QueryAnalysis:
    """Simple query characteristic analysis"""
    logger.info("query_analysis_started", query_length=len(query))
    
    query_lower = query.lower()
    words = query.split()
    
    # Determine intent
    if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
        intent = 'comparison'
    elif any(word in query_lower for word in ['relationship', 'correlation', 'association']):
        intent = 'relational'
    elif any(word in query_lower for word in ['analyze', 'pattern', 'trend', 'statistics']):
        intent = 'analytical'
    else:
        intent = 'factual'
    
    # Determine complexity
    complexity = 'complex' if len(words) > 15 else 'simple'
    
    # Count entities (simple heuristic)
    entity_indicators = ['patient', 'disease', 'condition', 'treatment', 'diagnosis']
    entity_count = sum(1 for indicator in entity_indicators if indicator in query_lower)
    
    # Check for relationships
    relationship_words = ['relationship', 'correlation', 'association', 'compare', 'versus']
    has_relationships = any(word in query_lower for word in relationship_words)
    
    result = QueryAnalysis(
        intent=intent,
        complexity=complexity,
        entity_count=max(entity_count, 1),
        has_relationships=has_relationships
    )
    
    logger.info("query_analysis_completed", 
               intent=intent, 
               complexity=complexity, 
               entity_count=result.entity_count,
               has_relationships=has_relationships)
    
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
    
    def register_tool(self, name: str, func: callable, description: str = ""):
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
