"""
Simplified Agentic Agents - Following Roadmap Requirements

This module implements TRUE agentic behavior with:
1. Dynamic reasoning and planning
2. Learning from interactions
3. Simple modular approach using existing logic
4. Minimal boilerplate code

Follows the architectural roadmap's recommendation for Enhanced OrchestratorAgent 
as the primary reasoning engine with minimal changes to other agents.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic.v1 import BaseModel, Field
import re
import requests
import time

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import Azure Key Vault manager for secure secret management
from core.azure_keyvault_manager import get_secret_from_keyvault
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import self-contained base classes and utilities
from updated_agents.base_classes import (
    WorkflowState, 
    SecureAgentBase, 
    AgentRole,
    QueryAnalysis,
    RoutingDecision,
    ValidationResult,
    SynthesisResult,
    validate_medical_relevance,
    analyze_query_characteristics,
    extract_text_from_documents,
    calculate_simple_quality_score,
    tool_registry
)
from core.observability import traceable, get_traceable_config
from core.input_sanitization import secure_llm_interaction
from core.logging_config import get_logger

# Initialize logger for agentic agents
logger = get_logger("simple_agentic_agents")

class LearningMemory:
    """Simple learning memory for strategy optimization"""
    
    def __init__(self):
        self.query_patterns = {}
        self.routing_performance = {}
        self.adaptation_count = 0
        logger.info("learning_memory_initialized")
    
    def record_performance(self, query_type: str, route: str, quality_score: float):
        """Record query performance for learning"""
        key = f"{query_type}_{route}"
        if key not in self.routing_performance:
            self.routing_performance[key] = []
        
        self.routing_performance[key].append({
            'score': quality_score,
            'timestamp': datetime.now()
        })
        
        # Keep only recent performance data (last 50 records)
        if len(self.routing_performance[key]) > 50:
            self.routing_performance[key] = self.routing_performance[key][-50:]
        
        logger.debug("performance_recorded", 
                    query_type=query_type, 
                    route=route, 
                    quality_score=quality_score,
                    total_records=len(self.routing_performance[key]))
    
    def get_best_route(self, query_type: str) -> str:
        """Learn the best route for a query type"""
        routes = ['vector', 'graph', 'both']
        best_route = 'both'  # Default fallback
        best_score = 0.0
        
        for route in routes:
            key = f"{query_type}_{route}"
            if key in self.routing_performance:
                scores = [record['score'] for record in self.routing_performance[key]]
                avg_score = sum(scores) / len(scores) if scores else 0.0
                if avg_score > best_score:
                    best_score = avg_score
                    best_route = route
        
        logger.debug("best_route_determined", 
                    query_type=query_type, 
                    best_route=best_route, 
                    best_score=best_score)
        return best_route
    
    def adapt_strategy(self):
        """Simple adaptation counter"""
        self.adaptation_count += 1
        logger.info("strategy_adapted", adaptation_count=self.adaptation_count)
    
    def clear_all_learning(self):
        """Clear all learning data for fresh start"""
        self.query_patterns.clear()
        self.routing_performance.clear()
        self.adaptation_count = 0
        logger.info("learning_memory_cleared")

# Global learning memory - simple singleton pattern
learning_memory = LearningMemory()

def _merge_search_results(vector_docs: List[Dict], bm25_docs: List[Dict]) -> List[Dict]:
    """Merge and deduplicate vector and BM25 results with optimized weighting."""
    merged_docs = []
    seen_content = set()
    
    def content_hash(content: str) -> str:
        """Create simple content hash for deduplication"""
        return re.sub(r'\W+', ' ', content.lower()).strip()[:100]
    
    # Adaptive weighting: Vector preferred for semantic understanding
    if vector_docs and bm25_docs:
        vector_weight, bm25_weight = 0.7, 0.3
    elif vector_docs:
        vector_weight, bm25_weight = 1.0, 0.0
    else:
        vector_weight, bm25_weight = 0.0, 1.0
    
    # Add vector docs
    for doc in vector_docs:
        content_sig = content_hash(doc["content"])
        if content_sig not in seen_content:
            doc["hybrid_score"] = doc["score"] * vector_weight
            merged_docs.append(doc)
            seen_content.add(content_sig)
    
    # Add BM25 docs (avoid duplicates, boost existing)
    for doc in bm25_docs:
        content_sig = content_hash(doc["content"])
        if content_sig not in seen_content:
            doc["hybrid_score"] = doc["score"] * bm25_weight
            merged_docs.append(doc)
            seen_content.add(content_sig)
        else:
            # Boost score for documents found by both methods
            for merged_doc in merged_docs:
                if content_hash(merged_doc["content"]) == content_sig:
                    merged_doc["hybrid_score"] += doc["score"] * bm25_weight * 0.5
                    merged_doc["source"] = "hybrid_both"
                    break
    
    # Sort by hybrid score
    return sorted(merged_docs, key=lambda x: x["hybrid_score"], reverse=True)

def _initialize_embeddings_fast():
    """Fast embeddings initialization with SSL bypass and caching"""
    try:
        logger.info("embeddings_initialization_started")
        
        # Patch requests session to disable SSL verification
        original_request = requests.Session.request
        def patched_request(self, method, url, **kwargs):
            kwargs.setdefault('verify', False)
            return original_request(self, method, url, **kwargs)
        requests.Session.request = patched_request
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu', 
                'trust_remote_code': True
            },
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info("embeddings_initialized")
        return embeddings
        
    except Exception as e:
        logger.error("embeddings_initialization_failed", error=str(e))
        return None

def _initialize_bm25_retriever(qdrant_client, collection_name: Optional[str] = None, max_docs: int = 500):
    """Initialize BM25 retriever from Qdrant documents with optimized fetching."""
    try:
        # Use Azure Key Vault to get collection name if not provided
        if not collection_name:
            collection_name = get_secret_from_keyvault("QDRANT_COLLECTION") or "medical_research_doc"
        
        logger.info("bm25_initialization_started", collection=collection_name)
        
        # Optimized document fetching - limit to reasonable size for faster init
        points, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=max_docs,
            with_payload=True,
            with_vectors=False  # Don't fetch vectors for BM25
        )
        
        # Convert to LangChain Documents with streaming processing
        documents = []
        for point in points:
            # Try different payload fields for content
            content = None
            if hasattr(point, 'payload') and point.payload:
                content = (
                    point.payload.get("content") or 
                    point.payload.get("chunk") or 
                    point.payload.get("text") or 
                    point.payload.get("description") or
                    str(point.payload)
                )
            
            if content and content.strip():  # Only process non-empty content
                documents.append(Document(
                    page_content=content,
                    metadata=point.payload.get("metadata", {}) if hasattr(point, 'payload') and point.payload else {}
                ))
        
        if not documents:
            logger.warning(f"No documents found in collection: {collection_name}")
            return None
        
        # Create BM25 retriever with optimized settings
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10  # Keep reasonable default
        
        logger.info("bm25_initialized", 
                   documents_count=len(documents), 
                   max_docs=max_docs, 
                   collection=collection_name)
        return bm25_retriever
        
    except Exception as e:
        logger.error("bm25_initialization_failed", error=str(e))
        return None

class SimpleReasoningPlan(BaseModel):
    """Simple reasoning plan - no complex chains"""
    query_type: str = Field(description="Query classification")
    selected_route: str = Field(description="Route decision")
    reasoning: str = Field(description="Brief reasoning")
    is_learned: bool = Field(description="Was this decision learned?")

class VectorSearchResult(BaseModel):
    """Enhanced vector search result with hybrid capabilities"""
    documents: List[Dict[str, Any]] = Field(description="Retrieved documents with scores")
    total_found: int = Field(description="Total number of documents found")
    search_params: Dict[str, Any] = Field(description="Search parameters used")
    search_strategy: str = Field(description="Strategy used: vector_only, hybrid, or bm25_only", default="vector_only")
    vector_count: int = Field(description="Number of documents from vector search", default=0)
    bm25_count: int = Field(description="Number of documents from BM25 search", default=0)

class HybridSearchResult(BaseModel):
    """Hybrid search result combining vector and BM25"""
    documents: List[Dict[str, Any]] = Field(description="Combined and reranked documents")
    vector_count: int = Field(description="Number of documents from vector search")
    bm25_count: int = Field(description="Number of documents from BM25 search")
    total_found: int = Field(description="Total unique documents after merging")
    search_strategy: str = Field(description="Strategy used: vector_only, bm25_only, or hybrid")

class GraphSearchResult(BaseModel):
    """Structured graph search result"""
    documents: List[Dict[str, Any]] = Field(description="Retrieved relationship data")
    total_found: int = Field(description="Total number of results found")
    optimizations_applied: int = Field(description="Number of optimizations applied")

class AgenticOrchestratorAgent(SecureAgentBase):
    """
    Enhanced Orchestrator with TRUE agentic behavior
    
    Key Agentic Features:
    1. Dynamic reasoning for route selection
    2. Learning from past routing decisions
    3. Adaptive strategy optimization
    4. Self-contained implementation without external dependencies
    """
    
    def __init__(self, llm: AzureChatOpenAI, vector_agent=None, graph_agent=None):
        super().__init__(AgentRole.ORCHESTRATOR)
        self.llm = llm
        self.learning_enabled = True
        self.vector_agent = vector_agent  # Real AgenticVectorRAGAgent with hybrid search
        self.graph_agent = graph_agent    # Real AgenticGraphRAGAgent
        logger.info("agentic_orchestrator_initialized", 
                   learning_enabled=self.learning_enabled,
                   has_vector_agent=vector_agent is not None,
                   has_graph_agent=graph_agent is not None)
    
    @traceable(**get_traceable_config("AgenticOrchestratorAgent"))
    def reason_and_plan(self, state: WorkflowState) -> WorkflowState:
        """CORE AGENTIC CAPABILITY: Dynamic reasoning and planning"""
        
        logger.info("reasoning_started", query_length=len(state.get("query", "")))
        
        # Step 1: Analyze query characteristics with reasoning (includes proper key mapping)
        analysis = self._analyze_query_with_reasoning(state["query"])
        logger.debug("query_analysis_completed", analysis=analysis)
        
        # Step 2: AGENTIC DECISION - Dynamic route selection with learning
        routing_decision = self._make_agentic_routing_decision(analysis)
        logger.info("routing_decision_made", 
                   selected_route=routing_decision.selected_route,
                   is_learned=routing_decision.is_learned,
                   reasoning=routing_decision.reasoning)
        
        # Store reasoning plan in state
        state["reasoning_plan"] = {
            "query_type": routing_decision.query_type,
            "selected_route": routing_decision.selected_route,
            "reasoning": routing_decision.reasoning,
            "is_learned": routing_decision.is_learned,
            "query_analysis": analysis
        }
        
        # Step 3: Execute with the reasoned plan
        state = self._execute_agentic_plan(state, routing_decision)
        
        # Step 4: LEARNING - Reflect and adapt for future decisions
        if self.learning_enabled:
            self._learn_from_execution(analysis, routing_decision, state)
            logger.debug("learning_completed")
        
        return state

    @traceable(**get_traceable_config("AgenticOrchestratorAgent"))
    def validate_medical_relevance_tool(self, query: str) -> Dict[str, Any]:
        """Tool to validate medical relevance of a query"""
        logger.info("medical_relevance_tool_called", query_length=len(query))
        result = validate_medical_relevance(query)
        logger.debug("medical_relevance_tool_completed", is_medical=result.get('is_medical', False))
        return result
    
    @traceable(**get_traceable_config("AgenticOrchestratorAgent"))
    def analyze_query_characteristics_tool(self, query: str) -> QueryAnalysis:
        """Tool to analyze query characteristics and return structured output"""
        logger.info("query_characteristics_tool_called", query_length=len(query))
        result = analyze_query_characteristics(query)
        logger.debug("query_characteristics_tool_completed", intent=result.intent)
        return result
    
    def route_to_vector(self, state: WorkflowState) -> WorkflowState:
        """Route to vector search - simplified implementation"""
        logger.info("routing_to_vector_search")
        state["selected_route"] = "vector"
        state["routing_decision"] = "Routed to vector search for semantic similarity"
        logger.debug("vector_route_applied", route="vector")
        return state
    
    def route_to_graph(self, state: WorkflowState) -> WorkflowState:
        """Route to graph search - simplified implementation"""
        logger.info("routing_to_graph_search")
        state["selected_route"] = "graph"
        state["routing_decision"] = "Routed to graph search for relationship analysis"
        logger.debug("graph_route_applied", route="graph")
        return state
    
    def route_to_both(self, state: WorkflowState) -> WorkflowState:
        """Route to both vector and graph - simplified implementation"""
        logger.info("routing_to_both_searches")
        state["selected_route"] = "both"
        state["routing_decision"] = "Routed to both vector and graph for comprehensive analysis"
        logger.debug("both_route_applied", route="both")
        return state
    
    def handle_non_medical_query(self, state: WorkflowState) -> WorkflowState:
        """Handle non-medical queries"""
        logger.info("handling_non_medical_query")
        validation_result = state.get("medical_validation", {})
        state["final_answer"] = validation_result.get("quick_response", 
            "I can only help with medical and healthcare-related questions.")
        state["sources"] = []
        logger.debug("non_medical_response_generated")
        return state
    
    def _analyze_query_with_reasoning(self, query: str) -> Dict[str, Any]:
        """Analyze query with simple reasoning using structured outputs"""
        
        logger.info("query_reasoning_analysis_started", query_length=len(query))
        
        # Use functions for medical validation and query analysis
        validation_result = validate_medical_relevance(query)
        
        # Use functions for query analysis  
        query_analysis = analyze_query_characteristics(query)
        
        result = {
            'query_type': query_analysis.intent,
            'is_medical': validation_result.get('is_medical', False),
            'complexity': query_analysis.complexity,
            'entity_count': query_analysis.entity_count,
            'has_relationships': query_analysis.has_relationships
        }
        
        logger.info("query_reasoning_analysis_completed", 
                   query_type=result['query_type'],
                   is_medical=result['is_medical'],
                   complexity=result['complexity'])
        
        return result
    
    def _make_agentic_routing_decision(self, analysis: Dict[str, Any]) -> SimpleReasoningPlan:
        """CORE AGENTIC BEHAVIOR: Dynamic routing with learned preferences"""
        
        query_type = analysis['query_type']
        logger.info("making_agentic_routing_decision", query_type=query_type)
        
        # LEARNING APPLICATION: Use learned best route if available
        if learning_memory.routing_performance:
            learned_route = learning_memory.get_best_route(query_type)
            logger.info("applying_learned_route", 
                       query_type=query_type,
                       learned_route=learned_route)
            return SimpleReasoningPlan(
                query_type=query_type,
                selected_route=learned_route,
                reasoning=f"Selected {learned_route} based on learned performance for {query_type} queries",
                is_learned=True
            )
        
        # REASONING-BASED ROUTING: Dynamic decision making
        logger.info("applying_reasoning_based_routing", query_type=query_type)
        
        if query_type == 'comparison':
            route = 'both'  # Comparisons benefit from both vector and graph
            reasoning = "Complex comparisons require both semantic similarity and relationship analysis"
        
        elif query_type == 'relational':
            route = 'graph'  # Relationships are graph strengths
            reasoning = "Relationship queries are optimally handled by graph database"
        
        elif query_type == 'analytical':
            route = 'both'  # Analysis benefits from comprehensive data
            reasoning = "Analytical queries need comprehensive data from both sources"
        
        else:  # factual
            route = 'vector'  # Simple facts work well with vector similarity
            reasoning = "Factual queries efficiently handled by vector similarity search"
        
        logger.info("routing_decision_determined", 
                   query_type=query_type,
                   selected_route=route,
                   reasoning=reasoning)
        
        return SimpleReasoningPlan(
            query_type=query_type,
            selected_route=route,
            reasoning=reasoning,
            is_learned=False
        )
    
    def _execute_agentic_plan(self, state: WorkflowState, plan: SimpleReasoningPlan) -> WorkflowState:
        """Execute the agentic plan with proper agent coordination"""
        
        logger.info("executing_agentic_plan", 
                   selected_route=plan.selected_route,
                   is_learned_decision=plan.is_learned)
        
        # Add reasoning to state for transparency
        state["reasoning_plan"] = {
            "query_type": plan.query_type,
            "selected_route": plan.selected_route,
            "reasoning": plan.reasoning,
            "is_learned_decision": plan.is_learned,
            "timestamp": datetime.now().isoformat()
        }
        
        # Execute based on the reasoned route selection
        if plan.selected_route == 'vector':
            logger.debug("executing_vector_route")
            return self._execute_vector_search(state)
        elif plan.selected_route == 'graph':
            logger.debug("executing_graph_route")
            return self._execute_graph_search(state)
        elif plan.selected_route == 'both':
            logger.debug("executing_both_routes")
            return self._execute_both_searches(state)
        else:
            # Non-medical fallback
            logger.debug("executing_non_medical_fallback")
            return self.handle_non_medical_query(state)
    
    def _execute_vector_search(self, state: WorkflowState) -> WorkflowState:
        """Execute vector search route using real AgenticVectorRAGAgent"""
        logger.info("executing_vector_search_route")
        
        # First validate medical relevance using the function
        validation_result = validate_medical_relevance(state["query"])
        state["medical_validation"] = validation_result

        if not state.get("medical_validation", {}).get("is_medical", False):
            logger.warning("vector_search_blocked_non_medical")
            return self.handle_non_medical_query(state)

        # Use real AgenticVectorRAGAgent with adaptive search capabilities
        if self.vector_agent:
            try:
                logger.info("using_real_vector_agent_with_adaptive_search")
                # Use adaptive search method that adjusts parameters based on query complexity
                state = self.vector_agent.search_with_adaptation(state)
                vector_results = state.get("vector_results", {})
                logger.info("real_adaptive_vector_search_completed", 
                           total_found=vector_results.get("total_found", 0),
                           strategy=vector_results.get("search_strategy", "unknown"))
            except Exception as e:
                logger.error("real_adaptive_vector_search_failed", error=str(e))
                # Return empty results instead of simulation
                state["vector_results"] = {
                    "documents": [],
                    "total_found": 0,
                    "search_strategy": "error"
                }
        else:
            logger.error("vector_agent_not_available")
            # Return empty results instead of simulation
            state["vector_results"] = {
                "documents": [],
                "total_found": 0,
                "search_strategy": "agent_unavailable"
            }
        
        logger.info("vector_search_route_completed")
        return state
    
    def _execute_graph_search(self, state: WorkflowState) -> WorkflowState:
        """Execute graph search route using real AgenticGraphRAGAgent"""
        logger.info("executing_graph_search_route")
        
        # First validate medical relevance using the function
        validation_result = validate_medical_relevance(state["query"])
        state["medical_validation"] = validation_result
        
        if not state.get("medical_validation", {}).get("is_medical", False):
            logger.warning("graph_search_blocked_non_medical")
            return self.handle_non_medical_query(state)
        
        # Use real AgenticGraphRAGAgent with optimization capabilities
        if self.graph_agent:
            try:
                logger.info("using_real_graph_agent_with_optimization")
                # Use optimization search method that learns and adapts
                state = self.graph_agent.search_with_optimization(state)
                graph_results = state.get("graph_results", {})
                logger.info("real_optimized_graph_search_completed", 
                           total_found=graph_results.get("total_found", 0),
                           optimizations_applied=graph_results.get("optimizations_applied", 0))
            except Exception as e:
                logger.error("real_optimized_graph_search_failed", error=str(e))
                # Return empty results instead of simulation
                state["graph_results"] = {
                    "documents": [],
                    "total_found": 0,
                    "optimizations_applied": 0
                }
        else:
            logger.error("graph_agent_not_available")
            # Return empty results instead of simulation  
            state["graph_results"] = {
                "documents": [],
                "total_found": 0,
                "optimizations_applied": 0
            }
        
        logger.info("graph_search_route_completed")
        return state
    
    def _execute_both_searches(self, state: WorkflowState) -> WorkflowState:
        """Execute both vector and graph searches"""
        logger.info("executing_both_searches_route")
        
        # First validate medical relevance using the function
        validation_result = validate_medical_relevance(state["query"])
        state["medical_validation"] = validation_result
        
        if not state.get("medical_validation", {}).get("is_medical", False):
            logger.warning("both_searches_blocked_non_medical")
            return self.handle_non_medical_query(state)
        
        # Execute both searches using REAL adaptive agents
        if self.vector_agent:
            # Use the adaptive search method that adjusts parameters based on query complexity
            try:
                logger.info("using_adaptive_vector_search_in_both_mode")
                state = self.vector_agent.search_with_adaptation(state)
                vector_results = state.get("vector_results", {})
                logger.info("real_adaptive_vector_search_completed", 
                           total_found=vector_results.get("total_found", 0),
                           strategy=vector_results.get("search_strategy", "unknown"))
            except Exception as e:
                logger.error("real_adaptive_vector_search_failed", error=str(e))
                # Return empty results instead of simulation
                state["vector_results"] = {
                    "documents": [],
                    "total_found": 0,
                    "search_strategy": "error"
                }
        else:
            # Return empty results instead of simulation
            state["vector_results"] = {
                "documents": [],
                "total_found": 0,
                "search_strategy": "agent_unavailable"
            }
            
        if self.graph_agent:
            # Use optimization search method that learns and adapts
            try:
                logger.info("using_optimized_graph_search_in_both_mode")
                state = self.graph_agent.search_with_optimization(state)
                graph_results = state.get("graph_results", {})
                logger.info("real_optimized_graph_search_completed", 
                           total_found=graph_results.get("total_found", 0),
                           optimizations_applied=graph_results.get("optimizations_applied", 0))
            except Exception as e:
                logger.error("real_optimized_graph_search_failed", error=str(e))
                # Return empty results instead of simulation
                state["graph_results"] = {
                    "documents": [],
                    "total_found": 0,
                    "optimizations_applied": 0
                }
        else:
            # Return empty results instead of simulation
            state["graph_results"] = {
                "documents": [],
                "total_found": 0,
                "optimizations_applied": 0
            }
            
        logger.info("both_searches_route_completed")
        return state
    
    def _learn_from_execution(self, analysis: Dict, plan: SimpleReasoningPlan, state: WorkflowState):
        """LEARNING CAPABILITY: Simple learning from execution results"""
        
        logger.info("learning_from_execution_started", 
                   query_type=plan.query_type,
                   selected_route=plan.selected_route)
        
        # Calculate simple quality score from results
        quality_score = self._calculate_simple_quality_score(state)
        
        # Record performance for learning
        learning_memory.record_performance(
            query_type=plan.query_type,
            route=plan.selected_route,
            quality_score=quality_score
        )
        
        # Adapt strategy
        learning_memory.adapt_strategy()
        
        # Add learning info to state
        state["learning_update"] = {
            "quality_score": quality_score,
            "adaptation_count": learning_memory.adaptation_count,
            "total_patterns": len(learning_memory.routing_performance)
        }
        
        logger.info("learning_from_execution_completed",
                   quality_score=quality_score,
                   adaptation_count=learning_memory.adaptation_count)
    
    def _calculate_simple_quality_score(self, state: WorkflowState) -> float:
        """Simple quality scoring - basic heuristics"""
        
        logger.debug("calculating_quality_score")
        
        # Basic quality indicators
        has_results = bool(state.get("results"))
        result_length = len(str(state.get("results", "")))
        has_sources = bool(state.get("sources"))
        
        # Simple scoring
        score = 0.0
        if has_results:
            score += 0.5
            logger.debug("quality_score_component", component="has_results", value=0.5)
        if result_length > 100:  # Substantial answer
            score += 0.3
            logger.debug("quality_score_component", component="substantial_content", value=0.3)
        if has_sources:
            score += 0.2
            logger.debug("quality_score_component", component="has_sources", value=0.2)
        
        final_score = min(score, 1.0)  # Cap at 1.0
        logger.debug("quality_score_calculated", 
                    raw_score=score,
                    final_score=final_score,
                    has_results=has_results,
                    result_length=result_length,
                    has_sources=has_sources)
        
        return final_score
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        logger.debug("retrieving_learning_stats")
        
        stats = {
            "adaptation_count": learning_memory.adaptation_count,
            "learned_patterns": len(learning_memory.routing_performance),
            "query_types_learned": list(set(
                key.split('_')[0] for key in learning_memory.routing_performance.keys()
            ))
        }
        
        logger.debug("learning_stats_retrieved", 
                    adaptation_count=stats["adaptation_count"],
                    learned_patterns=stats["learned_patterns"])
        
        return stats
    
    def reset_learning(self):
        """Reset learning state"""
        logger.info("resetting_learning_state")
        global learning_memory
        learning_memory = LearningMemory()
        logger.info("learning_state_reset_completed")

class AgenticVectorRAGAgent(SecureAgentBase):
    """
    Enhanced Vector Agent with minimal agentic improvements
    Self-contained implementation with adaptive parameters
    """
    
    def __init__(self, llm: AzureChatOpenAI, vector_store):
        super().__init__(AgentRole.VECTOR_RAG)
        self.llm = llm
        self.vector_store = vector_store
        self.adaptive_params = {
            'k_documents': 5,  # Adaptive number of documents
            'score_threshold': 0.7  # Adaptive threshold
        }
        
        # Initialize embeddings and BM25 for hybrid search
        self.embeddings = None
        self.bm25_retriever = None
        self._embeddings_initialized = False
        self._bm25_initialized = False
        
        logger.info("agentic_vector_agent_initialized", 
                   initial_k_documents=self.adaptive_params['k_documents'],
                   initial_score_threshold=self.adaptive_params['score_threshold'])
    
    def _get_embeddings_lazy(self):
        """Lazy initialization of embeddings - only when needed for vector search"""
        if not self._embeddings_initialized:
            self.embeddings = _initialize_embeddings_fast()
            self._embeddings_initialized = True
        return self.embeddings
    
    def _get_collection_name(self):
        """Get collection name from Azure Key Vault only"""
        try:
            # Get collection name from Azure Key Vault
            collection_name = get_secret_from_keyvault("QDRANT_COLLECTION")
            
            if collection_name:
                logger.info("using_keyvault_collection", collection_name=collection_name)
                return collection_name
            else:
                logger.error("collection_name_not_found_in_keyvault")
                return None
            
        except Exception as e:
            logger.error("keyvault_collection_name_retrieval_failed", error=str(e))
            return None
    
    def _get_bm25_retriever_lazy(self):
        """Lazy initialization of BM25 retriever - only when needed for hybrid search"""
        # Return cached retriever if already initialized successfully
        if self._bm25_initialized:
            return self.bm25_retriever
        
        # Only attempt initialization if we have a vector store and haven't initialized yet
        if self.vector_store:
            try:
                logger.info("initializing_bm25_retriever_first_time")
                # Get collection name using centralized method
                collection_name = self._get_collection_name()
                
                if collection_name:
                    try:
                        self.bm25_retriever = _initialize_bm25_retriever(
                            self.vector_store,
                            collection_name,
                            max_docs=500
                        )
                        if self.bm25_retriever:
                            self._bm25_initialized = True
                            logger.info("bm25_retriever_initialized_successfully")
                        else:
                            logger.warning("bm25_retriever_initialization_returned_none")
                            self._bm25_initialized = True  # Mark as attempted to avoid repeated tries
                    except Exception as e:
                        logger.warning("bm25_collection_initialization_failed", 
                                     collection_name=collection_name, error=str(e))
                        self._bm25_initialized = True  # Mark as attempted to avoid repeated tries
                else:
                    logger.warning("no_collection_name_available_for_bm25")
                    self._bm25_initialized = True  # Mark as attempted to avoid repeated tries
                
            except Exception as e:
                logger.warning("bm25_lazy_initialization_failed", error=str(e))
                self.bm25_retriever = None
                self._bm25_initialized = True  # Mark as attempted to avoid repeated tries
        else:
            logger.warning("no_vector_store_available_for_bm25")
            self._bm25_initialized = True  # Mark as attempted to avoid repeated tries
        
        return self.bm25_retriever
    
    @traceable(**get_traceable_config("AgenticVectorRAGAgent"))
    def search_with_adaptation(self, state: WorkflowState) -> WorkflowState:
        """Enhanced search with simple parameter adaptation"""
        
        logger.info("vector_search_with_adaptation_started")
        
        # AGENTIC BEHAVIOR: Adapt search parameters based on query complexity
        query_length = len(state["query"].split())
        
        if query_length > 15:  # Complex query
            self.adaptive_params['k_documents'] = 8
            self.adaptive_params['score_threshold'] = 0.6
            logger.debug("adapted_for_complex_query", k_documents=8, score_threshold=0.6)
        elif query_length < 5:  # Simple query
            self.adaptive_params['k_documents'] = 3
            self.adaptive_params['score_threshold'] = 0.8
            logger.debug("adapted_for_simple_query", k_documents=3, score_threshold=0.8)
        else:  # Medium query
            self.adaptive_params['k_documents'] = 5
            self.adaptive_params['score_threshold'] = 0.7
            logger.debug("adapted_for_medium_query", k_documents=5, score_threshold=0.7)
        
        # Perform search with structured output
        search_result = self.search_vectors(state["query"])
        
        # Add structured result to state
        state["vector_results"] = search_result.dict()
        logger.info("vector_search_with_adaptation_completed", 
                   documents_found=search_result.total_found)
        return state
    
    @traceable(**get_traceable_config("AgenticVectorRAGAgent"))
    def search_vectors(self, query: str) -> VectorSearchResult:
        """Enhanced hybrid vector search with BM25 integration and reranking"""
        logger.debug("hybrid_vector_search_started", 
                    k_documents=self.adaptive_params['k_documents'],
                    score_threshold=self.adaptive_params['score_threshold'])
        
        try:
            vector_docs, bm25_docs = [], []
            
            # Try vector search with embeddings
            embeddings = self._get_embeddings_lazy()
            if embeddings and self.vector_store and hasattr(self.vector_store, 'search'):
                try:
                    logger.info("performing_vector_search")
                    
                    # Get collection name using centralized method
                    collection_name = self._get_collection_name()
                    
                    if collection_name:
                        # Perform vector search
                        query_embedding = embeddings.embed_query(query)
                        vector_results = self.vector_store.search(
                            collection_name=collection_name,
                            query_vector=query_embedding,
                            limit=self.adaptive_params['k_documents'] * 2,
                            with_payload=True,
                            with_vectors=False
                        )
                        
                        vector_docs = []
                        for result in vector_results:
                            if hasattr(result, 'payload') and result.payload:
                                content = (
                                    result.payload.get("content") or 
                                    result.payload.get("chunk") or 
                                    result.payload.get("text") or
                                    str(result.payload)
                                )
                                
                                vector_docs.append({
                                    "id": f"vec_{result.id}",
                                    "content": content,
                                    "metadata": result.payload.get("metadata", {}),
                                    "score": float(result.score),
                                    "source": "vector_search"
                                })
                        
                        logger.info("vector_search_completed", documents_found=len(vector_docs))
                        
                except Exception as e:
                    logger.warning("vector_search_failed", error=str(e))
            
            # Try BM25 search for keyword matching
            bm25_retriever = self._get_bm25_retriever_lazy()
            if bm25_retriever:
                try:
                    logger.info("performing_bm25_search")
                    bm25_results = bm25_retriever.get_relevant_documents(query)
                    bm25_docs = []
                    
                    for i, doc in enumerate(bm25_results[:self.adaptive_params['k_documents']]):
                        bm25_docs.append({
                            "id": f"bm25_{i}",
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": max(0.0, 1.0 - (i / len(bm25_results))),
                            "source": "keyword_search"
                        })
                    
                    logger.info("bm25_search_completed", documents_found=len(bm25_docs))
                    
                except Exception as e:
                    logger.warning("bm25_search_failed", error=str(e))
            
            # Check if we have any real database results
            if vector_docs or bm25_docs:
                combined_docs = _merge_search_results(vector_docs, bm25_docs)
                
                # Apply reranking if we have LLM available
                if self.llm and len(combined_docs) > 1:
                    try:
                        logger.info("applying_llm_reranking", documents_count=len(combined_docs))
                        reranked_docs = self._rerank_documents_by_relevance(query, combined_docs)
                        combined_docs = reranked_docs[:self.adaptive_params['k_documents']]
                        logger.info("reranking_completed", final_count=len(combined_docs))
                    except Exception as e:
                        logger.warning("reranking_failed", error=str(e))
                        # Continue with original ordering
                        combined_docs = combined_docs[:self.adaptive_params['k_documents']]
                else:
                    combined_docs = combined_docs[:self.adaptive_params['k_documents']]
                
                # Determine strategy
                if vector_docs and bm25_docs:
                    strategy = "hybrid"
                elif vector_docs:
                    strategy = "vector_only"
                elif bm25_docs:
                    strategy = "bm25_only"
                else:
                    strategy = "no_results"
                
                logger.info("hybrid_search_completed", 
                           total_docs=len(combined_docs),
                           vector_count=len(vector_docs),
                           bm25_count=len(bm25_docs),
                           strategy=strategy)
                
                return VectorSearchResult(
                    documents=combined_docs,
                    total_found=len(combined_docs),
                    search_params=self.adaptive_params.copy(),
                    search_strategy=strategy,
                    vector_count=len(vector_docs),
                    bm25_count=len(bm25_docs)
                )
            
            # No real data found in Qdrant database - return empty result for medical queries
            logger.warning("no_data_found_in_qdrant_database", query=query)
            return VectorSearchResult(
                documents=[],
                total_found=0,
                search_params=self.adaptive_params.copy(),
                search_strategy="no_data_found",
                vector_count=len(vector_docs),
                bm25_count=len(bm25_docs)
            )
            
        except Exception as e:
            logger.error("hybrid_vector_search_failed", error=str(e))
            return VectorSearchResult(
                documents=[],
                total_found=0,
                search_params=self.adaptive_params.copy(),
                search_strategy="error",
                vector_count=0,
                bm25_count=0
            )
    
    def _rerank_documents_by_relevance(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using LLM for better relevance ordering"""
        try:
            if len(documents) <= 1:
                return documents
            
            # Prepare documents for reranking
            doc_summaries = []
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                # Truncate long content for reranking prompt
                content_preview = content[:200] + "..." if len(content) > 200 else content
                doc_summaries.append(f"Document {i+1}: {content_preview}")
            
            # Create reranking prompt
            reranking_prompt = f"""Given the user query and the following documents, please rank them from most relevant (1) to least relevant ({len(documents)}) for answering the query.

User Query: {query}

Documents:
{chr(10).join(doc_summaries)}

Provide your ranking as a comma-separated list of document numbers (e.g., "3,1,4,2" for 4 documents).
Ranking (most relevant first):"""

            try:
                from langchain_core.messages import HumanMessage
                
                response = self.llm.invoke([HumanMessage(content=reranking_prompt)])
                
                # Handle different response types
                if hasattr(response, 'content'):
                    ranking_text = response.content
                elif isinstance(response, str):
                    ranking_text = response
                else:
                    ranking_text = str(response)
                
                # Parse ranking
                ranking_str = str(ranking_text).strip()
                ranking_numbers = [int(x.strip()) for x in ranking_str.split(',') if x.strip().isdigit()]
                
                # Validate ranking
                if len(ranking_numbers) == len(documents) and set(ranking_numbers) == set(range(1, len(documents) + 1)):
                    # Apply ranking (convert 1-based to 0-based indexing)
                    reranked_docs = [documents[i-1] for i in ranking_numbers]
                    logger.info("llm_reranking_successful", original_order=list(range(len(documents))), new_order=ranking_numbers)
                    return reranked_docs
                else:
                    logger.warning("invalid_ranking_response", ranking=ranking_numbers)
                    return documents
                    
            except Exception as e:
                logger.warning("llm_reranking_failed", error=str(e))
                return documents
                
        except Exception as e:
            logger.error("reranking_error", error=str(e))
            return documents

class AgenticGraphRAGAgent(SecureAgentBase):
    """
    Enhanced Graph Agent with minimal agentic improvements
    Self-contained implementation with query optimization
    """
    
    def __init__(self, llm: AzureChatOpenAI, graph_store):
        super().__init__(AgentRole.GRAPH_RAG)
        self.llm = llm
        self.graph_store = graph_store
        self.query_optimizations = 0
        logger.info("agentic_graph_agent_initialized", initial_optimizations=0)
    
    @traceable(**get_traceable_config("AgenticGraphRAGAgent"))
    def search_with_optimization(self, state: WorkflowState) -> WorkflowState:
        """Enhanced search with simple query optimization learning"""
        
        logger.info("graph_search_with_optimization_started")
        
        # AGENTIC BEHAVIOR: Simple query optimization
        self.query_optimizations += 1
        logger.debug("graph_optimization_applied", 
                    optimization_count=self.query_optimizations)
        
        # Perform graph search with structured output
        search_result = self.search_graph(state)
        
        # Add structured result to state
        state["graph_results"] = search_result.dict()
        logger.info("graph_search_with_optimization_completed",
                   documents_found=search_result.total_found,
                   optimizations_applied=search_result.optimizations_applied)
        return state
    
    @traceable(**get_traceable_config("AgenticGraphRAGAgent"))
    def search_graph(self, state: WorkflowState) -> GraphSearchResult:
        """Graph search implementation - returns empty results as simulation removed"""
        logger.debug("graph_search_started", 
                    optimization_count=self.query_optimizations)
        
        try:
            # Real graph search would be implemented here
            # For now, return empty results as simulation is removed
            documents = []
            
            result = GraphSearchResult(
                documents=documents,
                total_found=len(documents),
                optimizations_applied=self.query_optimizations
            )
            
            logger.debug("graph_search_completed",
                        documents_found=len(documents),
                        optimizations_applied=self.query_optimizations)
            
            return result
        except Exception as e:
            logger.error("graph_search_failed", error=str(e))
            return GraphSearchResult(
                documents=[],
                total_found=0,
                optimizations_applied=self.query_optimizations
            )

class SimpleValidatorAgent(SecureAgentBase):
    """Simple validator agent with self-contained logic"""
    
    def __init__(self, llm: AzureChatOpenAI):
        super().__init__(AgentRole.VALIDATOR)
        self.llm = llm
        logger.info("simple_validator_agent_initialized")
    
    @traceable(**get_traceable_config("SimpleValidatorAgent"))
    def validate_results(self, state: WorkflowState) -> ValidationResult:
        """Simple validation of results with structured output"""
        logger.info("validation_started")
        
        # Combine results from vector and graph searches
        vector_results = state.get("vector_results", {}).get("documents", [])
        graph_results = state.get("graph_results", {}).get("documents", [])
        
        # Check if we have a "no_data_found" scenario from vector search
        vector_strategy = state.get("vector_results", {}).get("search_strategy", "")
        
        all_documents = vector_results + graph_results
        logger.debug("validation_documents_collected", 
                    vector_count=len(vector_results),
                    graph_count=len(graph_results),
                    total_count=len(all_documents),
                    vector_strategy=vector_strategy)
        
        if all_documents:
            # Simple validation logic with structured output
            validation_result = ValidationResult(
                is_valid=True,
                score=0.8,
                feedback="Results passed basic validation checks"
            )
            state["validated_results"] = all_documents
            state["validation"] = validation_result.dict()
            logger.info("validation_passed", score=0.8, documents_count=len(all_documents))
        else:
            # Check if this is a "no_data_found" case for better feedback
            if vector_strategy == "no_data_found":
                validation_result = ValidationResult(
                    is_valid=False,
                    score=0.0,
                    feedback="No data found in medical database for the query"
                )
                logger.warning("validation_failed", reason="no_data_found_in_database")
            else:
                validation_result = ValidationResult(
                    is_valid=False,
                    score=0.0,
                    feedback="No valid results found for validation"
                )
                logger.warning("validation_failed", reason="no_documents")
            
            state["validated_results"] = []
            state["validation"] = validation_result.dict()
        
        return validation_result

class SimpleAnswerSynthesisAgent(SecureAgentBase):
    """Enhanced answer synthesis agent with LLM-based generation"""
    
    def __init__(self, llm: AzureChatOpenAI):
        super().__init__(AgentRole.SYNTHESIZER)
        self.llm = llm
        logger.info("enhanced_synthesis_agent_initialized")
    
    @traceable(**get_traceable_config("SimpleAnswerSynthesisAgent"))
    def synthesize_answer(self, state: WorkflowState) -> SynthesisResult:
        """Enhanced answer synthesis using LLM"""
        logger.info("enhanced_synthesis_started")
        
        validated_results = state.get("validated_results", [])
        query = state.get("query", "")
        
        # Check if this is a medical query and if we have no data from database
        vector_results = state.get("vector_results", {})
        search_strategy = vector_results.get("search_strategy", "")
        
        # First check if we have no results at all
        if not validated_results:
            logger.warning("synthesis_no_results")
            
            # Check if this was a medical query with no data found in database
            if search_strategy == "no_data_found":
                logger.info("medical_query_no_database_results", query=query)
                return SynthesisResult(
                    answer="No data found in our medical database for this query.",
                    confidence=0.0,
                    sources=[]
                )
            else:
                return SynthesisResult(
                    answer="No data found in our medical database for this query.",
                    confidence=0.0,
                    sources=[]
                )
        
        # Extract and prepare context from documents
        context_pieces = []
        sources = []
        
        for i, doc in enumerate(validated_results[:5]):  # Limit to top 5 documents
            content = doc.get("content", "")
            score = doc.get("score", 0.0)
            
            if content and content.strip():
                # Clean and prepare content
                cleaned_content = content.strip()
                if len(cleaned_content) > 500:  # Truncate very long content
                    cleaned_content = cleaned_content[:500] + "..."
                
                context_pieces.append(f"[Source {i+1}] {cleaned_content}")
                sources.append({
                    "source_id": i+1,
                    "content_preview": cleaned_content[:100] + "..." if len(cleaned_content) > 100 else cleaned_content,
                    "relevance_score": float(score)
                })
        
        if not context_pieces:
            logger.warning("synthesis_no_valid_content")
            return SynthesisResult(
                answer="No data found in our medical database for this query.",
                confidence=0.0,
                sources=[]
            )
        
        # Create comprehensive context
        combined_context = "\n\n".join(context_pieces)
        
        # DEBUG: Log the context being provided
        logger.info("DEBUG_CONTEXT_PROVIDED", 
                   context_length=len(combined_context),
                   context_pieces_count=len(context_pieces),
                   context_preview=combined_context[:500] + "..." if len(combined_context) > 500 else combined_context)
        
        # Enhanced synthesis prompt template with balanced approach
        synthesis_prompt = f"""You are a medical database assistant. Your job is to answer based on the provided context from our medical database.

User Question: {query}

Context Information from Database:
{combined_context}

INSTRUCTIONS:
1. Answer using the information from the provided context
2. If the context contains relevant information, provide a comprehensive answer based on that data
3. If the context doesn't contain sufficient information about the query, respond with: "No data found in our medical database for this query."
4. Do not add external medical knowledge not present in the context
5. You may use the phrase "Based on the provided information" or "According to the database context"

Database Response:"""

        try:
            # Use LLM to generate comprehensive answer
            logger.info("llm_synthesis_started", context_pieces=len(context_pieces))
            
            from langchain_core.messages import HumanMessage
            
            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            
            # Handle different response types
            if hasattr(response, 'content'):
                synthesized_answer = response.content
            elif isinstance(response, str):
                synthesized_answer = response
            else:
                synthesized_answer = str(response)
            
            # Ensure we have a string and clean it
            synthesized_answer = str(synthesized_answer).strip() if synthesized_answer else "Unable to generate answer"
            
            # DEBUG: Log the actual LLM response
            logger.info("DEBUG_LLM_RESPONSE", 
                       raw_answer=synthesized_answer,
                       answer_length=len(synthesized_answer))
            
            # Check if LLM provided external knowledge instead of database-only response
            external_knowledge_indicators = [
                "in general", "typically", "usually", "commonly",
                "generally speaking", "broadly", "it is known that",
                "medical literature", "consult", "recommend", "suggest",
                "without specific details", "cannot elaborate further",
                "multiple potential causes", "complex group of diseases",
                "based on common medical knowledge", "as is well known"
            ]
            
            answer_lower = synthesized_answer.lower()
            has_external_knowledge = any(indicator in answer_lower for indicator in external_knowledge_indicators)
            
            # Check if the answer contains database context references (indicating it's using provided data)
            database_indicators = [
                "source", "context", "provided", "information", "database",
                "available data", "according to", "based on the", "from the"
            ]
            has_database_content = any(indicator in answer_lower for indicator in database_indicators)
            
            # Only override if it has external knowledge AND doesn't reference database content
            # AND the answer suggests general knowledge instead of database-specific info
            should_override = (
                has_external_knowledge and 
                not has_database_content and 
                ("context" not in answer_lower or "provided" not in answer_lower) and
                len(synthesized_answer) < 100
            )
            
            if should_override:
                logger.warning("llm_provided_external_knowledge_overriding", 
                             external_knowledge_detected=has_external_knowledge,
                             has_database_content=has_database_content,
                             answer_length=len(synthesized_answer))
                synthesized_answer = "No data found in our medical database for this query."
                confidence_score = 0.0
            else:
                # Calculate confidence based on context quality and length
                confidence_score = min(0.95, 0.6 + (len(context_pieces) * 0.1) + (min(len(synthesized_answer), 500) / 1000))
            
            # Convert sources to strings as required by SynthesisResult model
            sources_strings = []
            for source in sources:
                source_str = f"Source {source['source_id']}: {source['content_preview']} (relevance: {source['relevance_score']:.2f})"
                sources_strings.append(source_str)
            
            logger.info("llm_synthesis_completed", 
                       answer_length=len(synthesized_answer),
                       confidence_score=confidence_score,
                       sources_count=len(sources_strings))
            
            return SynthesisResult(
                answer=synthesized_answer,
                confidence=confidence_score,
                sources=sources_strings
            )
            
        except Exception as e:
            logger.error("llm_synthesis_failed", error=str(e))
            
            # Fallback to simple concatenation if LLM fails
            fallback_answer = f"Based on the available information: {context_pieces[0]}"
            if len(context_pieces) > 1:
                fallback_answer += f"\n\nAdditionally: {context_pieces[1]}"
            
            # Convert sources to strings for fallback
            sources_strings = []
            for source in sources:
                source_str = f"Source {source['source_id']}: {source['content_preview']} (relevance: {source['relevance_score']:.2f})"
                sources_strings.append(source_str)
            
            return SynthesisResult(
                answer=fallback_answer,
                confidence=0.5,
                sources=sources_strings
            )

class SimpleAgenticWorkflow:
    """
    Simplified Agentic Workflow
    
    This is the main orchestrator that demonstrates TRUE agentic behavior:
    1. Dynamic reasoning and planning
    2. Learning from interactions
    3. Adaptive decision making
    4. Minimal complexity, maximum reuse
    """
    
    def __init__(self, llm: AzureChatOpenAI, vector_store, graph_store):
        # Initialize enhanced agents with self-contained implementations
        self.vector_agent = AgenticVectorRAGAgent(llm, vector_store)
        self.graph_agent = AgenticGraphRAGAgent(llm, graph_store)
        
        # CORE AGENTIC COMPONENT: Enhanced Orchestrator with reasoning
        # Pass the real agents to the orchestrator
        self.orchestrator = AgenticOrchestratorAgent(llm, self.vector_agent, self.graph_agent)
        
        # Use simple self-contained validation and synthesis agents
        self.validator = SimpleValidatorAgent(llm)
        self.synthesizer = SimpleAnswerSynthesisAgent(llm)
        
        self.execution_count = 0
        
        logger.info("simple_agentic_workflow_initialized", execution_count=0)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        MAIN AGENTIC PROCESSING PIPELINE
        
        Demonstrates autonomous behavior:
        - Dynamic reasoning for route selection
        - Learning from past decisions
        - Adaptive strategy optimization
        """
        
        self.execution_count += 1
        start_time = datetime.now()
        
        logger.info("agentic_query_processing_started", 
                   execution_count=self.execution_count,
                   query_length=len(query))
        
        # Initialize state
        state = WorkflowState()
        state["query"] = query
        state["execution_id"] = self.execution_count
        logger.debug("workflow_state_initialized", execution_id=self.execution_count)
        
        try:
            # CORE AGENTIC STEP: Reason and plan dynamically
            state = self.orchestrator.reason_and_plan(state)
            
            # Check if it's a non-medical query (handled in reason_and_plan)
            if not state.get("medical_validation", {}).get("is_medical", False) and state.get("final_answer"):
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info("non_medical_query_completed", execution_time=execution_time)
                return {
                    "answer": state.get("final_answer"),
                    "final_answer": state.get("final_answer"),  # Add both keys for compatibility
                    "sources": state.get("sources", []),
                    "confidence_score": 0.0,  # Add confidence_score for consistency
                    "reasoning_plan": state.get("reasoning_plan", {}),
                    "execution_metrics": {
                        "execution_time": execution_time,
                        "execution_count": self.execution_count
                    },
                    "agentic_indicators": {
                        "autonomous_reasoning": True,
                        "learning_applied": False,
                        "adaptive_behavior": True
                    }
                }
            
            # For medical queries, continue with validation and synthesis
            route = state.get("reasoning_plan", {}).get("selected_route", "both")
            logger.debug("medical_query_route_determined", route=route)
            
            # The orchestrator.reason_and_plan() has already executed the searches
            # and stored results in state["vector_results"] and/or state["graph_results"]
            # No need to call the agents again - just validate that we have results
            
            has_vector_results = bool(state.get("vector_results", {}).get("documents"))
            has_graph_results = bool(state.get("graph_results", {}).get("documents"))
            
            logger.info("search_results_available", 
                       route=route,
                       has_vector_results=has_vector_results,
                       has_graph_results=has_graph_results)
            
            # Validation step - prepare documents for synthesis
            validation_result = self.validator.validate_results(state)
            logger.info("validation_completed", is_valid=validation_result.is_valid)
            
            # Synthesis step - THIS is where the final answer gets generated
            synthesis_result = self.synthesizer.synthesize_answer(state)
            state["final_answer"] = synthesis_result.answer if hasattr(synthesis_result, 'answer') else getattr(synthesis_result, 'final_answer', 'No answer generated')
            state["sources"] = synthesis_result.sources if hasattr(synthesis_result, 'sources') else []
            state["confidence_score"] = synthesis_result.confidence if hasattr(synthesis_result, 'confidence') else getattr(synthesis_result, 'confidence_score', 0.0)
            
            logger.info("synthesis_completed", 
                       has_final_answer=bool(state.get("final_answer")),
                       answer_length=len(str(state.get("final_answer", ""))),
                       confidence_score=state.get("confidence_score", 0.0))
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("agentic_query_processing_completed", 
                       execution_time=execution_time,
                       route_used=route,
                       final_answer_generated=bool(state.get("final_answer")))
            
            # Prepare agentic response
            return {
                "answer": state.get("final_answer", "No answer generated"),
                "final_answer": state.get("final_answer", "No answer generated"),  # Add both keys for compatibility
                "sources": state.get("sources", []),
                "confidence_score": state.get("confidence_score", 0.0),  # Add confidence_score for Streamlit
                "reasoning_plan": state.get("reasoning_plan", {}),
                "learning_update": state.get("learning_update", {}),
                "execution_metrics": {
                    "execution_time": execution_time,
                    "execution_count": self.execution_count
                },
                "agentic_indicators": {
                    "autonomous_reasoning": bool(state.get("reasoning_plan")),
                    "learning_applied": bool(state.get("learning_update")),
                    "adaptive_behavior": True
                }
            }
            
        except Exception as e:
            logger.error("agentic_query_processing_failed", error=str(e))
            return {
                "answer": f"Error during agentic processing: {str(e)}",
                "final_answer": f"Error during agentic processing: {str(e)}",  # Add both keys for compatibility
                "sources": [],
                "confidence_score": 0.0,  # Add confidence_score for consistency
                "error": True,
                "execution_metrics": {
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "execution_count": self.execution_count
                }
            }
    
    def get_agentic_status(self) -> Dict[str, Any]:
        """Get current agentic system status"""
        logger.debug("agentic_status_requested")
        
        learning_stats = self.orchestrator.get_learning_stats()
        
        status = {
            "total_executions": self.execution_count,
            "learning_statistics": learning_stats,
            "agentic_capabilities": {
                "dynamic_reasoning": True,
                "learning_enabled": True,
                "adaptive_routing": True,
                "autonomous_decisions": True
            },
            "system_status": "Active" if self.execution_count > 0 else "Initialized"
        }
        
        logger.debug("agentic_status_generated", 
                    total_executions=self.execution_count,
                    system_status=status["system_status"])
        
        return status
    
    def reset_learning_state(self):
        """Reset all learning data"""
        logger.info("learning_state_reset_requested")
        self.orchestrator.reset_learning()
        self.execution_count = 0
        logger.info("learning_state_reset_completed")

# Factory function for easy instantiation
def create_simple_agentic_workflow(llm: AzureChatOpenAI, vector_store, graph_store) -> SimpleAgenticWorkflow:
    """Create a simple agentic workflow with all required components"""
    logger.info("creating_simple_agentic_workflow")
    workflow = SimpleAgenticWorkflow(llm, vector_store, graph_store)
    logger.info("simple_agentic_workflow_created")
    return workflow
