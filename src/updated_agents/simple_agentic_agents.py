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
from core.input_sanitization import (
    detect_prompt_injection,
    sanitize_user_input,
    validate_llm_output,
    create_secure_prompt_template,
    secure_llm_interaction,
    MEDICAL_VALIDATION_TEMPLATE,
    QUERY_ANALYSIS_TEMPLATE,
    ENTITY_EXTRACTION_TEMPLATE,
    DOCUMENT_RERANKING_TEMPLATE,
    SYNTHESIS_TEMPLATE
)
from core.logging_config import get_logger

# Initialize logger for agentic agents
logger = get_logger("simple_agentic_agents")

# Security templates for agentic LLM interactions
SEARCH_STRATEGY_TEMPLATE = """
You are an expert search strategy advisor for medical document retrieval. Analyze the user query and recommend the optimal search strategy.

Available Strategies:
1. vector_only: Best for semantic similarity, conceptual queries, finding documents with similar meaning
2. bm25_only: Best for exact keyword matching, specific terms, definitions, precise terminology
3. hybrid: Best for complex queries requiring both semantic understanding and keyword precision

<USER_QUERY>{user_query}</USER_QUERY>

Instructions:
1. Analyze the query characteristics (semantic vs keyword focus, complexity, specificity)
2. Consider the query type and medical domain context
3. Recommend ONE strategy: vector_only, bm25_only, or hybrid
4. Provide brief reasoning (max 100 characters)

Response format:
Strategy: [strategy_name]
Reasoning: [brief explanation]
"""

RELATIONSHIP_REASONING_TEMPLATE = """
You are an expert in medical knowledge graphs and relationship analysis. Analyze the query to determine the optimal graph traversal strategy.

<USER_QUERY>{user_query}</USER_QUERY>

Available Strategies:
1. breadth_first: Best for finding many related concepts, comparisons, broad relationships
2. depth_first: Best for causal chains, detailed pathways, deep connections
3. targeted: Best for specific relationships, focused queries, direct connections

Consider:
- Relationship types: causal, hierarchical, associative, temporal, comparative
- Query complexity and medical domain context
- Optimal traversal depth and breadth requirements

Response format:
Strategy: [breadth_first|depth_first|targeted]
Reasoning: [brief explanation max 100 chars]
Confidence: [high|medium|low]
"""

ENHANCED_SYNTHESIS_TEMPLATE = """
You are a medical database assistant. Your job is to answer based on the provided context from our medical database.

<USER_QUERY>{user_query}</USER_QUERY>

Context Information from Database:
{combined_context}

INSTRUCTIONS:
1. Answer using the information from the provided context
2. If the context contains relevant information, provide a comprehensive answer based on that data
3. If the context doesn't contain sufficient information about the query, respond with: "No data found in our medical database for this query."
4. Do not add external medical knowledge not present in the context
5. You may use the phrase "Based on the provided information" or "According to the database context"

Database Response:
"""

ENHANCED_DOCUMENT_RERANKING_TEMPLATE = """
Given the user query and the following documents, please rank them from most relevant (1) to least relevant for answering the query.

<USER_QUERY>{user_query}</USER_QUERY>

<DOCUMENTS>
{document_list}
</DOCUMENTS>

Provide your ranking as a comma-separated list of document numbers (e.g., "3,1,4,2" for 4 documents).
Ranking (most relevant first):
"""

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
    1. Dynamic reasoning for route selection and goal-setting
    2. Learning from past routing decisions and outcomes
    3. Adaptive strategy optimization with context awareness
    4. Tool orchestration with dynamic invocation
    5. Goal-oriented planning with sub-goal decomposition
    6. Inter-agent communication and coordination
    """
    
    def __init__(self, llm: AzureChatOpenAI, vector_agent=None, graph_agent=None):
        super().__init__(AgentRole.ORCHESTRATOR)
        self.llm = llm
        self.learning_enabled = True
        self.vector_agent = vector_agent  # Real AgenticVectorRAGAgent with hybrid search
        self.graph_agent = graph_agent    # Real AgenticGraphRAGAgent
        
        # Enhanced agentic capabilities
        self.available_tools = self._initialize_tool_registry()
        self.execution_goals = []
        self.current_context = {}
        self.reasoning_history = []
        
        logger.info("agentic_orchestrator_initialized", 
                   learning_enabled=self.learning_enabled,
                   has_vector_agent=vector_agent is not None,
                   has_graph_agent=graph_agent is not None,
                   available_tools=len(self.available_tools))
    
    def _initialize_tool_registry(self) -> Dict[str, Any]:
        """Initialize dynamic tool registry for agentic behavior"""
        tools = {
            "medical_validation": {
                "function": validate_medical_relevance,
                "description": "Validate if query is medical/healthcare related",
                "usage_count": 0,
                "success_rate": 1.0
            },
            "query_analysis": {
                "function": analyze_query_characteristics,
                "description": "Analyze query complexity and characteristics",
                "usage_count": 0,
                "success_rate": 1.0
            },
            "vector_search": {
                "function": self._execute_vector_search,
                "description": "Perform semantic vector search",
                "usage_count": 0,
                "success_rate": 1.0
            },
            "graph_search": {
                "function": self._execute_graph_search,
                "description": "Perform relationship-based graph search",
                "usage_count": 0,
                "success_rate": 1.0
            },
            "hybrid_search": {
                "function": self._execute_both_searches,
                "description": "Perform both vector and graph searches",
                "usage_count": 0,
                "success_rate": 1.0
            }
        }
        logger.info("tool_registry_initialized", tool_count=len(tools))
        return tools
    
    def _set_execution_goals(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Set execution goals based on query analysis - TRUE AGENTIC PLANNING"""
        goals = []
        
        # Primary goal: Answer the user's question
        primary_goal = {
            "id": "primary_answer",
            "description": f"Provide comprehensive answer to: {query[:100]}...",
            "priority": 1,
            "status": "pending",
            "sub_goals": []
        }
        
        # Add sub-goals based on query complexity and type
        if analysis.get('complexity') == 'complex':
            primary_goal["sub_goals"].extend([
                {
                    "id": "validate_medical",
                    "description": "Validate medical relevance",
                    "tool": "medical_validation",
                    "status": "pending"
                },
                {
                    "id": "analyze_complexity",
                    "description": "Analyze query characteristics",
                    "tool": "query_analysis", 
                    "status": "pending"
                },
                {
                    "id": "search_data",
                    "description": "Search for relevant information",
                    "tool": "hybrid_search",
                    "status": "pending"
                }
            ])
        else:
            primary_goal["sub_goals"].extend([
                {
                    "id": "validate_medical",
                    "description": "Validate medical relevance",
                    "tool": "medical_validation",
                    "status": "pending"
                },
                {
                    "id": "search_data",
                    "description": "Search for relevant information",
                    "tool": "vector_search",
                    "status": "pending"
                }
            ])
        
        goals.append(primary_goal)
        
        # Learning goal: Improve future performance
        learning_goal = {
            "id": "learning",
            "description": "Learn from execution for future improvement",
            "priority": 2,
            "status": "pending",
            "sub_goals": []
        }
        goals.append(learning_goal)
        
        self.execution_goals = goals
        logger.info("execution_goals_set", goal_count=len(goals), primary_subgoals=len(primary_goal["sub_goals"]))
        return goals
    
    def _invoke_tool_dynamically(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Dynamically invoke tools based on reasoning - CORE AGENTIC CAPABILITY"""
        if tool_name not in self.available_tools:
            logger.error("tool_not_available", tool_name=tool_name)
            return {"success": False, "error": f"Tool {tool_name} not available"}
        
        tool_info = self.available_tools[tool_name]
        tool_function = tool_info["function"]
        
        try:
            logger.info("invoking_tool_dynamically", tool_name=tool_name, kwargs=list(kwargs.keys()))
            
            # Update usage statistics
            tool_info["usage_count"] += 1
            
            # Invoke the tool
            result = tool_function(**kwargs)
            
            # Update success rate
            tool_info["success_rate"] = (tool_info["success_rate"] * (tool_info["usage_count"] - 1) + 1.0) / tool_info["usage_count"]
            
            logger.info("tool_invoked_successfully", 
                       tool_name=tool_name,
                       usage_count=tool_info["usage_count"],
                       success_rate=tool_info["success_rate"])
            
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error("tool_invocation_failed", tool_name=tool_name, error=str(e))
            
            # Update success rate
            current_failures = tool_info["usage_count"] * (1 - tool_info["success_rate"])
            new_failures = current_failures + 1
            tool_info["success_rate"] = 1 - (new_failures / tool_info["usage_count"])
            
            return {"success": False, "error": str(e)}
    
    def _update_reasoning_history(self, step: str, reasoning: str, outcome: Any):
        """Track reasoning process for learning and transparency"""
        reasoning_entry = {
            "step": step,
            "reasoning": reasoning,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        }
        
        self.reasoning_history.append(reasoning_entry)
        
        # Keep only recent history (last 20 entries)
        if len(self.reasoning_history) > 20:
            self.reasoning_history = self.reasoning_history[-20:]
        
        logger.debug("reasoning_history_updated", 
                    step=step,
                    total_entries=len(self.reasoning_history))
    
    @traceable(**get_traceable_config("AgenticOrchestratorAgent"))
    def reason_and_plan(self, state: WorkflowState) -> WorkflowState:
        """ENHANCED AGENTIC CAPABILITY: Dynamic reasoning, goal-setting, and execution planning"""
        
        logger.info("enhanced_reasoning_started", query_length=len(state.get("query", "")))
        
        # Step 1: Deep Query Analysis with Reasoning
        query = state["query"]
        analysis = self._analyze_query_with_reasoning(query)
        
        self._update_reasoning_history(
            "query_analysis",
            f"Analyzed query characteristics: {analysis['query_type']}, complexity: {analysis['complexity']}",
            analysis
        )
        
        # Step 2: AGENTIC GOAL SETTING - Decompose query into actionable goals
        execution_goals = self._set_execution_goals(query, analysis)
        state["execution_goals"] = execution_goals
        
        self._update_reasoning_history(
            "goal_setting",
            f"Set {len(execution_goals)} main goals with {len(execution_goals[0]['sub_goals'])} sub-goals",
            execution_goals
        )
        
        # Step 3: AGENTIC DECISION MAKING - Dynamic route selection with goal-awareness
        routing_decision = self._make_enhanced_agentic_routing_decision(analysis, execution_goals)
        
        self._update_reasoning_history(
            "route_decision",
            f"Selected route: {routing_decision.selected_route} based on {routing_decision.reasoning}",
            routing_decision
        )
        
        # Step 4: AGENTIC EXECUTION - Execute plan with goal tracking
        state = self._execute_agentic_plan_with_goals(state, routing_decision, execution_goals)
        
        # Step 5: AGENTIC LEARNING - Reflect and adapt based on goal achievement
        if self.learning_enabled:
            self._learn_from_goal_execution(analysis, routing_decision, state, execution_goals)
        
        # Step 6: Store comprehensive reasoning context
        state["reasoning_plan"] = {
            "query_type": routing_decision.query_type,
            "selected_route": routing_decision.selected_route,
            "reasoning": routing_decision.reasoning,
            "is_learned": routing_decision.is_learned,
            "query_analysis": analysis,
            "execution_goals": execution_goals,
            "reasoning_history": self.reasoning_history[-5:],  # Last 5 reasoning steps
            "tool_performance": {tool: info["success_rate"] for tool, info in self.available_tools.items()}
        }
        
        logger.info("enhanced_reasoning_completed", 
                   goals_achieved=sum(1 for goal in execution_goals if goal.get("status") == "completed"),
                   reasoning_steps=len(self.reasoning_history))
        
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
    
    def _make_enhanced_agentic_routing_decision(self, analysis: Dict[str, Any], execution_goals: List[Dict]) -> SimpleReasoningPlan:
        """ENHANCED AGENTIC BEHAVIOR: Goal-aware routing with dynamic tool selection"""
        
        query_type = analysis['query_type']
        complexity = analysis.get('complexity', 'simple')
        
        logger.info("making_enhanced_agentic_routing_decision", 
                   query_type=query_type,
                   complexity=complexity,
                   goal_count=len(execution_goals))
        
        # ADVANCED LEARNING APPLICATION: Consider tool performance history
        if learning_memory.routing_performance:
            learned_route = learning_memory.get_best_route(query_type)
            
            # Check if learned route aligns with current goals
            primary_goal = execution_goals[0] if execution_goals else {}
            sub_goal_tools = {sg.get("tool", "") for sg in primary_goal.get("sub_goals", [])}
            
            # Validate learned route against current tool requirements
            if self._validate_route_against_goals(learned_route, sub_goal_tools):
                logger.info("applying_validated_learned_route", 
                           query_type=query_type,
                           learned_route=learned_route,
                           aligned_with_goals=True)
                return SimpleReasoningPlan(
                    query_type=query_type,
                    selected_route=learned_route,
                    reasoning=f"Selected {learned_route} based on learned performance for {query_type} queries, validated against current goals",
                    is_learned=True
                )
        
        # ENHANCED REASONING-BASED ROUTING: Consider complexity and goals
        reasoning_factors = []
        
        if complexity == 'complex':
            reasoning_factors.append("Complex query detected - comprehensive search needed")
            if query_type == 'comparison':
                route = 'both'
                reasoning_factors.append("Comparisons require both semantic and relationship analysis")
            elif query_type == 'analytical':
                route = 'both'
                reasoning_factors.append("Analytical queries need multi-modal data synthesis")
            else:
                route = 'hybrid_search'
                reasoning_factors.append("Complex queries benefit from hybrid approach")
        else:
            reasoning_factors.append("Simple query detected - focused search appropriate")
            if query_type == 'relational':
                route = 'graph'
                reasoning_factors.append("Relationship focus ideal for graph database")
            elif analysis.get('has_relationships', False):
                route = 'both'
                reasoning_factors.append("Relationship elements detected - need both searches")
            else:
                route = 'vector'
                reasoning_factors.append("Factual query - vector search sufficient")
        
        # Consider available tool performance
        if route in self.available_tools:
            tool_success_rate = self.available_tools[route]["success_rate"]
            if tool_success_rate < 0.7:  # Low success rate
                reasoning_factors.append(f"Adjusting route due to {route} tool performance ({tool_success_rate:.2f})")
                route = 'both'  # Fallback to comprehensive search
        
        combined_reasoning = "; ".join(reasoning_factors)
        
        logger.info("enhanced_routing_decision_determined", 
                   query_type=query_type,
                   complexity=complexity,
                   selected_route=route,
                   reasoning_factors=len(reasoning_factors))
        
        return SimpleReasoningPlan(
            query_type=query_type,
            selected_route=route,
            reasoning=combined_reasoning,
            is_learned=False
        )
    
    def _validate_route_against_goals(self, route: str, sub_goal_tools: set) -> bool:
        """Validate if learned route aligns with current execution goals"""
        route_tools = {
            'vector': {'vector_search'},
            'graph': {'graph_search'},
            'both': {'vector_search', 'graph_search', 'hybrid_search'},
            'hybrid_search': {'hybrid_search'}
        }
        
        available_tools = route_tools.get(route, set())
        required_tools = sub_goal_tools - {'medical_validation', 'query_analysis'}  # Exclude validation tools
        
        alignment = bool(required_tools.intersection(available_tools)) or not required_tools
        
        logger.debug("route_goal_validation",
                    route=route,
                    available_tools=list(available_tools),
                    required_tools=list(required_tools),
                    alignment=alignment)
        
        return alignment
    
    def _execute_agentic_plan_with_goals(self, state: WorkflowState, plan: SimpleReasoningPlan, goals: List[Dict]) -> WorkflowState:
        """Execute plan with goal tracking and dynamic adaptation"""
        
        logger.info("executing_agentic_plan_with_goals", 
                   selected_route=plan.selected_route,
                   goal_count=len(goals))
        
        # Track goal execution
        primary_goal = goals[0] if goals else {}
        sub_goals = primary_goal.get("sub_goals", [])
        
        # Execute sub-goals in order, adapting as needed
        for sub_goal in sub_goals:
            if sub_goal["status"] == "pending":
                tool_name = sub_goal.get("tool", "")
                
                self._update_reasoning_history(
                    "sub_goal_execution",
                    f"Executing sub-goal: {sub_goal['description']} using tool: {tool_name}",
                    sub_goal
                )
                
                # Dynamic tool invocation based on sub-goal
                if tool_name == "medical_validation":
                    result = self._invoke_tool_dynamically("medical_validation", query=state["query"])
                    if result["success"]:
                        state["medical_validation"] = result["result"]
                        sub_goal["status"] = "completed"
                    else:
                        sub_goal["status"] = "failed"
                        sub_goal["error"] = result["error"]
                
                elif tool_name == "query_analysis":
                    result = self._invoke_tool_dynamically("query_analysis", query=state["query"])
                    if result["success"]:
                        state["query_characteristics"] = result["result"]
                        sub_goal["status"] = "completed"
                    else:
                        sub_goal["status"] = "failed"
                        sub_goal["error"] = result["error"]
                
                elif tool_name in ["vector_search", "graph_search", "hybrid_search"]:
                    result = self._invoke_tool_dynamically(tool_name, state=state)
                    if result["success"]:
                        state = result["result"]  # Updated state with search results
                        sub_goal["status"] = "completed"
                    else:
                        sub_goal["status"] = "failed"
                        sub_goal["error"] = result["error"]
                        
                        # ADAPTIVE BEHAVIOR: Try alternative if primary fails
                        if tool_name == "vector_search":
                            logger.info("vector_search_failed_trying_alternative")
                            alt_result = self._invoke_tool_dynamically("hybrid_search", state=state)
                            if alt_result["success"]:
                                state = alt_result["result"]
                                sub_goal["status"] = "completed_alternative"
                                sub_goal["alternative_tool"] = "hybrid_search"
        
        # Update goal status
        completed_sub_goals = sum(1 for sg in sub_goals if sg["status"].startswith("completed"))
        if completed_sub_goals == len(sub_goals):
            primary_goal["status"] = "completed"
        elif completed_sub_goals > 0:
            primary_goal["status"] = "partially_completed"
        else:
            primary_goal["status"] = "failed"
        
        # Update state with goal tracking
        state["goal_execution_summary"] = {
            "total_goals": len(goals),
            "total_sub_goals": len(sub_goals),
            "completed_sub_goals": completed_sub_goals,
            "primary_goal_status": primary_goal["status"],
            "execution_adaptations": sum(1 for sg in sub_goals if "alternative_tool" in sg)
        }
        
        logger.info("agentic_plan_execution_completed",
                   primary_goal_status=primary_goal["status"],
                   completed_sub_goals=completed_sub_goals,
                   total_sub_goals=len(sub_goals))
        
        return state
    
    def _learn_from_goal_execution(self, analysis: Dict, plan: SimpleReasoningPlan, state: WorkflowState, goals: List[Dict]):
        """ENHANCED LEARNING: Learn from goal achievement and execution patterns"""
        
        logger.info("enhanced_learning_from_goal_execution_started", 
                   query_type=plan.query_type,
                   selected_route=plan.selected_route)
        
        # Calculate enhanced quality score based on goal achievement
        quality_score = self._calculate_goal_aware_quality_score(state, goals)
        
        # Record performance with goal context
        learning_memory.record_performance(
            query_type=plan.query_type,
            route=plan.selected_route,
            quality_score=quality_score
        )
        
        # Learn from goal execution patterns
        goal_summary = state.get("goal_execution_summary", {})
        
        # Track tool effectiveness for future tool selection
        for goal in goals:
            for sub_goal in goal.get("sub_goals", []):
                tool_name = sub_goal.get("tool", "")
                if tool_name in self.available_tools and sub_goal.get("status", "").startswith("completed"):
                    # Boost success rate for effective tools
                    tool_info = self.available_tools[tool_name]
                    current_rate = tool_info["success_rate"]
                    # Slightly boost successful tools
                    tool_info["success_rate"] = min(1.0, current_rate + 0.01)
                    
                    logger.debug("tool_effectiveness_learned",
                               tool_name=tool_name,
                               new_success_rate=tool_info["success_rate"])
        
        # Adapt strategy based on execution adaptations
        adaptations = goal_summary.get("execution_adaptations", 0)
        if adaptations > 0:
            learning_memory.adapt_strategy()
            logger.info("strategy_adapted_due_to_execution_adaptations", adaptations=adaptations)
        
        # Add enhanced learning info to state
        state["enhanced_learning_update"] = {
            "goal_aware_quality_score": quality_score,
            "adaptation_count": learning_memory.adaptation_count,
            "total_patterns": len(learning_memory.routing_performance),
            "tool_success_rates": {tool: info["success_rate"] for tool, info in self.available_tools.items()},
            "execution_adaptations": adaptations
        }
        
        logger.info("enhanced_learning_completed",
                   goal_aware_quality_score=quality_score,
                   adaptation_count=learning_memory.adaptation_count,
                   execution_adaptations=adaptations)
    
    def _calculate_goal_aware_quality_score(self, state: WorkflowState, goals: List[Dict]) -> float:
        """Calculate quality score based on goal achievement and result quality"""
        
        logger.debug("calculating_goal_aware_quality_score")
        
        # Base quality from results
        base_score = self._calculate_simple_quality_score(state)
        
        # Goal achievement bonus
        goal_summary = state.get("goal_execution_summary", {})
        total_sub_goals = goal_summary.get("total_sub_goals", 1)
        completed_sub_goals = goal_summary.get("completed_sub_goals", 0)
        
        goal_achievement_ratio = completed_sub_goals / total_sub_goals if total_sub_goals > 0 else 0
        goal_bonus = goal_achievement_ratio * 0.3  # Up to 30% bonus for goal achievement
        
        # Adaptation penalty/bonus
        adaptations = goal_summary.get("execution_adaptations", 0)
        adaptation_factor = 0.1 if adaptations > 0 else 0  # Small bonus for successful adaptation
        
        final_score = min(1.0, base_score + goal_bonus + adaptation_factor)
        
        logger.debug("goal_aware_quality_calculated",
                    base_score=base_score,
                    goal_achievement_ratio=goal_achievement_ratio,
                    goal_bonus=goal_bonus,
                    adaptation_factor=adaptation_factor,
                    final_score=final_score)
        
        return final_score
    
    def _make_agentic_routing_decision(self, analysis: Dict[str, Any]) -> SimpleReasoningPlan:
        """ORIGINAL AGENTIC BEHAVIOR: Dynamic routing with learned preferences (kept for compatibility)"""
        
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
    Enhanced Vector Agent with TRUE agentic improvements
    
    Agentic Features:
    1. Dynamic parameter adaptation based on query analysis
    2. Tool selection reasoning (vector vs hybrid vs BM25)
    3. Self-monitoring and performance tracking
    4. Adaptive strategy learning from search outcomes
    5. Goal-oriented search planning
    """
    
    def __init__(self, llm: AzureChatOpenAI, vector_store):
        super().__init__(AgentRole.VECTOR_RAG)
        self.llm = llm
        self.vector_store = vector_store
        
        # Enhanced agentic capabilities
        self.adaptive_params = {
            'k_documents': 5,  # Adaptive number of documents
            'score_threshold': 0.7,  # Adaptive threshold
            'search_strategy': 'auto'  # auto, vector_only, hybrid, bm25_only
        }
        
        # Agentic performance tracking
        self.search_history = []
        self.strategy_performance = {
            'vector_only': {'attempts': 0, 'success_rate': 1.0, 'avg_relevance': 0.0},
            'hybrid': {'attempts': 0, 'success_rate': 1.0, 'avg_relevance': 0.0},
            'bm25_only': {'attempts': 0, 'success_rate': 1.0, 'avg_relevance': 0.0}
        }
        
        # Initialize embeddings and BM25 for hybrid search
        self.embeddings = None
        self.bm25_retriever = None
        self._embeddings_initialized = False
        self._bm25_initialized = False
        
        logger.info("enhanced_agentic_vector_agent_initialized", 
                   initial_k_documents=self.adaptive_params['k_documents'],
                   initial_score_threshold=self.adaptive_params['score_threshold'],
                   search_strategy=self.adaptive_params['search_strategy'])
    
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
        """ENHANCED AGENTIC SEARCH: Dynamic strategy selection and parameter adaptation"""
        
        logger.info("enhanced_agentic_vector_search_started")
        
        query = state["query"]
        
        # AGENTIC REASONING: Analyze query to determine optimal search strategy
        search_strategy = self._reason_about_search_strategy(query)
        
        # AGENTIC ADAPTATION: Adapt parameters based on query characteristics and learning
        self._adapt_search_parameters(query, search_strategy)
        
        # AGENTIC EXECUTION: Execute search with chosen strategy
        search_result = self._execute_reasoned_search(query, search_strategy)
        
        # AGENTIC LEARNING: Learn from search outcome
        self._learn_from_search_outcome(query, search_strategy, search_result)
        
        # Add structured result to state
        state["vector_results"] = search_result.dict()
        
        logger.info("enhanced_agentic_vector_search_completed", 
                   documents_found=search_result.total_found,
                   strategy_used=search_strategy)
        return state
    
    def _reason_about_search_strategy(self, query: str) -> str:
        """ENHANCED AGENTIC REASONING: Use LLM to determine optimal search strategy with security"""
        
        logger.info("llm_based_search_strategy_reasoning_started", query_length=len(query))
        
        # First check learned performance for quick decisions
        best_strategy = self._get_best_learned_strategy()
        if best_strategy and self.strategy_performance[best_strategy]['attempts'] > 5:
            logger.info("using_learned_strategy", strategy=best_strategy)
            return best_strategy
        
        # Use LLM for sophisticated reasoning with proper security
        try:
            logger.info("invoking_llm_for_strategy_reasoning")
            
            # Use secure LLM interaction with proper template
            validated_content = secure_llm_interaction(
                llm=self.llm,
                template=SEARCH_STRATEGY_TEMPLATE,
                user_input=query
            )
            
            if validated_content:
                # Parse LLM response securely
                strategy, reasoning = self._parse_llm_strategy_response(validated_content)
                
                if strategy in ['vector_only', 'bm25_only', 'hybrid']:
                    logger.info("llm_strategy_reasoning_successful", 
                               strategy=strategy,
                               reasoning=reasoning)
                    return strategy
                else:
                    logger.warning("llm_returned_invalid_strategy", 
                                 strategy=strategy,
                                 falling_back_to_heuristic=True)
            else:
                logger.warning("llm_strategy_reasoning_failed", 
                             error="No validated content returned")
        
        except Exception as e:
            logger.error("llm_strategy_reasoning_error", error=str(e))
        
        # Fallback to enhanced heuristic reasoning if LLM fails
        return self._fallback_heuristic_strategy_reasoning(query)
    
    def _prepare_strategy_context(self) -> str:
        """Prepare performance context for LLM reasoning"""
        
        context_parts = []
        
        for strategy, perf in self.strategy_performance.items():
            if perf['attempts'] > 0:
                context_parts.append(
                    f"{strategy}: {perf['attempts']} attempts, "
                    f"{perf['success_rate']:.2f} success rate, "
                    f"{perf['avg_relevance']:.2f} avg relevance"
                )
        
        if not context_parts:
            return "No performance history available"
        
        return "; ".join(context_parts)
    
    def _parse_llm_strategy_response(self, response: str) -> tuple[str, str]:
        """Securely parse LLM response for strategy and reasoning"""
        
        # Sanitize response
        response = response.strip()[:500]  # Limit length
        
        strategy = "hybrid"  # Default fallback
        reasoning = "LLM reasoning applied"
        
        try:
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.lower().startswith('strategy:'):
                    strategy_part = line.split(':', 1)[1].strip().lower()
                    if strategy_part in ['vector_only', 'bm25_only', 'hybrid']:
                        strategy = strategy_part
                
                elif line.lower().startswith('reasoning:'):
                    reasoning_part = line.split(':', 1)[1].strip()
                    if reasoning_part and len(reasoning_part) < 200:  # Reasonable length
                        reasoning = reasoning_part
            
            logger.debug("llm_response_parsed", 
                        strategy=strategy,
                        reasoning=reasoning[:50])
        
        except Exception as e:
            logger.warning("llm_response_parsing_failed", error=str(e))
        
        return strategy, reasoning
    
    def _fallback_heuristic_strategy_reasoning(self, query: str) -> str:
        """Enhanced fallback heuristic reasoning when LLM is unavailable"""
        
        logger.info("using_enhanced_heuristic_reasoning")
        
        query_lower = query.lower()
        words = query.split()
        word_count = len(words)
        
        reasoning_factors = []
        
        # Enhanced heuristic analysis
        semantic_indicators = [
            'meaning', 'similar', 'like', 'conceptual', 'understanding', 
            'explain', 'describe', 'what is', 'how does', 'why',
            'compare', 'difference', 'relationship', 'association'
        ]
        
        keyword_indicators = [
            'specific', 'exact', 'term', 'definition', 'name',
            'called', 'refers to', 'known as', 'acronym',
            'code', 'classification', 'type', 'category'
        ]
        
        complex_indicators = [
            'analyze', 'comprehensive', 'detailed', 'thorough',
            'multiple', 'various', 'different', 'all',
            'compare and contrast', 'pros and cons'
        ]
        
        # Calculate indicator scores
        semantic_score = sum(1 for indicator in semantic_indicators if indicator in query_lower)
        keyword_score = sum(1 for indicator in keyword_indicators if indicator in query_lower)
        complex_score = sum(1 for indicator in complex_indicators if indicator in query_lower)
        
        # Decision logic with weighted scoring
        if complex_score > 0 or word_count > 15:
            strategy = 'hybrid'
            reasoning_factors.append(f"Complex query detected (complexity score: {complex_score}, words: {word_count})")
        
        elif semantic_score > keyword_score and semantic_score > 1:
            strategy = 'vector_only'
            reasoning_factors.append(f"Strong semantic focus (semantic: {semantic_score} vs keyword: {keyword_score})")
        
        elif keyword_score > semantic_score and keyword_score > 0:
            strategy = 'bm25_only'
            reasoning_factors.append(f"Keyword-focused query (keyword: {keyword_score} vs semantic: {semantic_score})")
        
        elif word_count < 5:
            strategy = 'vector_only'
            reasoning_factors.append(f"Short query - semantic search preferred ({word_count} words)")
        
        else:
            strategy = 'hybrid'
            reasoning_factors.append("Balanced query - comprehensive search approach")
        
        # Consider performance history as tiebreaker
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            if perf['attempts'] > 0 and perf['success_rate'] < 0.5:
                reasoning_factors.append(f"Adjusting due to poor {strategy} performance ({perf['success_rate']:.2f})")
                strategy = 'hybrid'  # Default to comprehensive approach
        
        reasoning = "; ".join(reasoning_factors)
        
        logger.info("enhanced_heuristic_strategy_determined", 
                   strategy=strategy,
                   semantic_score=semantic_score,
                   keyword_score=keyword_score,
                   complex_score=complex_score,
                   reasoning=reasoning)
        
        return strategy
    
    def _get_best_learned_strategy(self) -> Optional[str]:
        """Get the best performing strategy from learning history"""
        
        if not any(perf['attempts'] > 0 for perf in self.strategy_performance.values()):
            return None
        
        best_strategy = None
        best_score = 0.0
        
        for strategy, performance in self.strategy_performance.items():
            if performance['attempts'] > 0:
                # Combine success rate and relevance for overall score
                combined_score = (performance['success_rate'] * 0.7) + (performance['avg_relevance'] * 0.3)
                if combined_score > best_score:
                    best_score = combined_score
                    best_strategy = strategy
        
        logger.debug("best_learned_strategy_determined",
                    strategy=best_strategy,
                    score=best_score)
        
        return best_strategy
    
    def _adapt_search_parameters(self, query: str, strategy: str):
        """AGENTIC ADAPTATION: Adapt search parameters based on strategy and query"""
        
        original_params = self.adaptive_params.copy()
        
        query_complexity = len(query.split())
        
        # Strategy-specific adaptations
        if strategy == 'vector_only':
            self.adaptive_params['k_documents'] = min(8, max(3, query_complexity // 3))
            self.adaptive_params['score_threshold'] = 0.75
        elif strategy == 'bm25_only':
            self.adaptive_params['k_documents'] = min(10, max(5, query_complexity // 2))
            self.adaptive_params['score_threshold'] = 0.6
        elif strategy == 'hybrid':
            self.adaptive_params['k_documents'] = min(12, max(6, query_complexity // 2))
            self.adaptive_params['score_threshold'] = 0.65
        
        # Learning-based fine-tuning
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            if perf['attempts'] > 2:
                if perf['success_rate'] < 0.7:
                    # Lower threshold if strategy has been failing
                    self.adaptive_params['score_threshold'] *= 0.9
                    self.adaptive_params['k_documents'] = min(15, int(self.adaptive_params['k_documents'] * 1.2))
                elif perf['avg_relevance'] > 0.8:
                    # Tighten parameters if getting very relevant results
                    self.adaptive_params['score_threshold'] *= 1.05
        
        self.adaptive_params['search_strategy'] = strategy
        
        logger.info("search_parameters_adapted",
                   strategy=strategy,
                   original_params=original_params,
                   adapted_params=self.adaptive_params)
    
    def _execute_reasoned_search(self, query: str, strategy: str) -> VectorSearchResult:
        """Execute search with the reasoned strategy"""
        
        logger.info("executing_reasoned_search", strategy=strategy)
        
        if strategy == 'vector_only':
            return self._search_vector_only(query)
        elif strategy == 'bm25_only':
            return self._search_bm25_only(query)
        else:  # hybrid
            return self.search_vectors(query)  # Use existing hybrid method
    
    def _search_vector_only(self, query: str) -> VectorSearchResult:
        """Vector-only search strategy"""
        
        logger.info("executing_vector_only_search")
        
        try:
            vector_docs = []
            
            embeddings = self._get_embeddings_lazy()
            if embeddings and self.vector_store and hasattr(self.vector_store, 'search'):
                collection_name = self._get_collection_name()
                
                if collection_name:
                    query_embedding = embeddings.embed_query(query)
                    vector_results = self.vector_store.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=self.adaptive_params['k_documents'] * 2,
                        with_payload=True,
                        with_vectors=False
                    )
                    
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
            
            # Apply score threshold filtering
            filtered_docs = [
                doc for doc in vector_docs 
                if doc["score"] >= self.adaptive_params['score_threshold']
            ][:self.adaptive_params['k_documents']]
            
            return VectorSearchResult(
                documents=filtered_docs,
                total_found=len(filtered_docs),
                search_params=self.adaptive_params.copy(),
                search_strategy="vector_only",
                vector_count=len(filtered_docs),
                bm25_count=0
            )
            
        except Exception as e:
            logger.error("vector_only_search_failed", error=str(e))
            return VectorSearchResult(
                documents=[],
                total_found=0,
                search_params=self.adaptive_params.copy(),
                search_strategy="vector_only_error",
                vector_count=0,
                bm25_count=0
            )
    
    def _search_bm25_only(self, query: str) -> VectorSearchResult:
        """BM25-only search strategy"""
        
        logger.info("executing_bm25_only_search")
        
        try:
            bm25_docs = []
            
            bm25_retriever = self._get_bm25_retriever_lazy()
            if bm25_retriever:
                bm25_results = bm25_retriever.get_relevant_documents(query)
                
                for i, doc in enumerate(bm25_results[:self.adaptive_params['k_documents']]):
                    bm25_docs.append({
                        "id": f"bm25_{i}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": max(0.0, 1.0 - (i / len(bm25_results))),
                        "source": "keyword_search"
                    })
            
            return VectorSearchResult(
                documents=bm25_docs,
                total_found=len(bm25_docs),
                search_params=self.adaptive_params.copy(),
                search_strategy="bm25_only",
                vector_count=0,
                bm25_count=len(bm25_docs)
            )
            
        except Exception as e:
            logger.error("bm25_only_search_failed", error=str(e))
            return VectorSearchResult(
                documents=[],
                total_found=0,
                search_params=self.adaptive_params.copy(),
                search_strategy="bm25_only_error",
                vector_count=0,
                bm25_count=0
            )
    
    def _learn_from_search_outcome(self, query: str, strategy: str, result: VectorSearchResult):
        """AGENTIC LEARNING: Learn from search outcomes to improve future decisions"""
        
        logger.info("learning_from_search_outcome", 
                   strategy=strategy,
                   documents_found=result.total_found)
        
        # Calculate relevance score (simple heuristic)
        relevance_score = min(1.0, result.total_found / max(1, self.adaptive_params['k_documents']))
        
        # Update strategy performance
        if strategy in self.strategy_performance:
            perf = self.strategy_performance[strategy]
            perf['attempts'] += 1
            
            # Update success rate (successful if we got results)
            success = result.total_found > 0
            current_successes = perf['success_rate'] * (perf['attempts'] - 1)
            new_success_rate = (current_successes + (1.0 if success else 0.0)) / perf['attempts']
            perf['success_rate'] = new_success_rate
            
            # Update average relevance
            current_relevance_total = perf['avg_relevance'] * (perf['attempts'] - 1)
            perf['avg_relevance'] = (current_relevance_total + relevance_score) / perf['attempts']
            
            logger.info("strategy_performance_updated",
                       strategy=strategy,
                       attempts=perf['attempts'],
                       success_rate=perf['success_rate'],
                       avg_relevance=perf['avg_relevance'])
        
        # Store search history for pattern recognition
        search_entry = {
            "query": query[:100],  # Store truncated query
            "strategy": strategy,
            "result_count": result.total_found,
            "relevance_score": relevance_score,
            "timestamp": datetime.now().isoformat()
        }
        
        self.search_history.append(search_entry)
        
        # Keep only recent history (last 50 searches)
        if len(self.search_history) > 50:
            self.search_history = self.search_history[-50:]
        
        logger.debug("search_history_updated", 
                    total_entries=len(self.search_history))
    
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
        """Rerank documents using LLM for better relevance ordering with security"""
        try:
            if len(documents) <= 1:
                return documents
            
            # Prepare documents for reranking
            doc_summaries = []
            for i, doc in enumerate(documents):
                content = doc.get("content", "")
                # Truncate long content for reranking prompt
                content_preview = sanitize_user_input(content[:200]) + "..." if len(content) > 200 else sanitize_user_input(content)
                doc_summaries.append(f"Document {i+1}: {content_preview}")
            
            # Create document list for template
            document_list = "\n".join(doc_summaries)
            
            try:
                # Use secure LLM interaction with proper template
                validated_content = secure_llm_interaction(
                    llm=self.llm,
                    template=ENHANCED_DOCUMENT_RERANKING_TEMPLATE,
                    user_input=query,
                    document_list=document_list
                )
                
                if validated_content:
                    # Parse ranking
                    ranking_str = str(validated_content).strip()
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
                else:
                    logger.warning("llm_reranking_failed", error="No validated content returned")
                    return documents
                    
            except Exception as e:
                logger.warning("llm_reranking_failed", error=str(e))
                return documents
                
        except Exception as e:
            logger.error("reranking_error", error=str(e))
            return documents

class AgenticGraphRAGAgent(SecureAgentBase):
    """
    Enhanced Graph Agent with TRUE agentic improvements
    
    Agentic Features:
    1. Dynamic query optimization based on graph patterns
    2. Relationship reasoning and path planning
    3. Adaptive search depth and breadth
    4. Learning from graph traversal outcomes
    5. Goal-oriented relationship discovery
    """
    
    def __init__(self, llm: AzureChatOpenAI, graph_store):
        super().__init__(AgentRole.GRAPH_RAG)
        self.llm = llm
        self.graph_store = graph_store
        
        # Enhanced agentic capabilities
        self.query_optimizations = 0
        self.relationship_patterns = {}
        self.search_strategies = {
            'breadth_first': {'attempts': 0, 'success_rate': 1.0, 'avg_depth': 0},
            'depth_first': {'attempts': 0, 'success_rate': 1.0, 'avg_depth': 0},
            'targeted': {'attempts': 0, 'success_rate': 1.0, 'avg_depth': 0}
        }
        
        # Adaptive parameters
        self.adaptive_params = {
            'max_depth': 3,
            'max_breadth': 10,
            'relationship_threshold': 0.7,
            'search_strategy': 'auto'
        }
        
        logger.info("enhanced_agentic_graph_agent_initialized", 
                   initial_optimizations=0,
                   search_strategies=list(self.search_strategies.keys()))
    
    @traceable(**get_traceable_config("AgenticGraphRAGAgent"))
    def search_with_optimization(self, state: WorkflowState) -> WorkflowState:
        """ENHANCED AGENTIC SEARCH: Relationship reasoning and adaptive graph traversal"""
        
        logger.info("enhanced_agentic_graph_search_started")
        
        query = state["query"]
        
        # AGENTIC REASONING: Analyze query for relationship patterns
        relationship_strategy = self._reason_about_relationships(query)
        
        # AGENTIC OPTIMIZATION: Optimize search parameters based on query and learning
        self._optimize_graph_parameters(query, relationship_strategy)
        
        # AGENTIC EXECUTION: Execute graph search with reasoned strategy
        search_result = self._execute_reasoned_graph_search(query, relationship_strategy)
        
        # AGENTIC LEARNING: Learn from graph traversal outcomes
        self._learn_from_graph_outcome(query, relationship_strategy, search_result)
        
        # Add structured result to state
        state["graph_results"] = search_result.dict()
        
        logger.info("enhanced_agentic_graph_search_completed",
                   documents_found=search_result.total_found,
                   optimizations_applied=search_result.optimizations_applied,
                   strategy_used=relationship_strategy)
        return state
    
    def _reason_about_relationships(self, query: str) -> str:
        """ENHANCED AGENTIC REASONING: Use LLM to analyze relationship patterns in query"""
        
        logger.info("llm_relationship_reasoning_started", query_length=len(query))
        
        # Check learned patterns first for quick decisions
        if len(self.relationship_patterns) > 5:
            best_pattern = self._get_best_learned_pattern(query)
            if best_pattern:
                logger.info("using_learned_relationship_pattern", pattern=best_pattern)
                return best_pattern
        
        # Use LLM for sophisticated relationship reasoning
        try:
            logger.info("invoking_llm_for_relationship_reasoning")
            
            # Use secure LLM interaction with proper template
            validated_content = secure_llm_interaction(
                llm=self.llm,
                template=RELATIONSHIP_REASONING_TEMPLATE,
                user_input=query
            )
            
            if validated_content:
                # Parse LLM response securely
                strategy, reasoning, confidence = self._parse_llm_relationship_response(validated_content)
                
                if strategy in ['breadth_first', 'depth_first', 'targeted']:
                    logger.info("llm_relationship_reasoning_successful",
                               strategy=strategy,
                               reasoning=reasoning,
                               confidence=confidence)
                    return strategy
                else:
                    logger.warning("llm_returned_invalid_relationship_strategy",
                                 strategy=strategy,
                                 falling_back_to_heuristic=True)
            else:
                logger.warning("llm_relationship_reasoning_failed",
                             error="No validated content returned")
        
        except Exception as e:
            logger.error("llm_relationship_reasoning_error", error=str(e))
        
        # Fallback to enhanced heuristic reasoning
        return self._fallback_heuristic_relationship_reasoning(query)
    
    def _get_strategy_performance_summary(self) -> str:
        """Get performance summary for LLM context"""
        
        summary_parts = []
        for strategy, perf in self.search_strategies.items():
            if perf['attempts'] > 0:
                summary_parts.append(f"{strategy}: {perf['success_rate']:.2f} success")
        
        return "; ".join(summary_parts) if summary_parts else "No history"
    
    def _parse_llm_relationship_response(self, response: str) -> tuple[str, str, str]:
        """Securely parse LLM response for relationship strategy"""
        
        # Sanitize response
        response = response.strip()[:300]
        
        strategy = "targeted"  # Default fallback
        reasoning = "LLM relationship reasoning applied"
        confidence = "medium"
        
        try:
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.lower().startswith('strategy:'):
                    strategy_part = line.split(':', 1)[1].strip().lower()
                    if strategy_part in ['breadth_first', 'depth_first', 'targeted']:
                        strategy = strategy_part
                
                elif line.lower().startswith('reasoning:'):
                    reasoning_part = line.split(':', 1)[1].strip()
                    if reasoning_part and len(reasoning_part) < 150:
                        reasoning = reasoning_part
                
                elif line.lower().startswith('confidence:'):
                    confidence_part = line.split(':', 1)[1].strip().lower()
                    if confidence_part in ['high', 'medium', 'low']:
                        confidence = confidence_part
            
            logger.debug("llm_relationship_response_parsed", 
                        strategy=strategy,
                        confidence=confidence)
        
        except Exception as e:
            logger.warning("llm_relationship_response_parsing_failed", error=str(e))
        
        return strategy, reasoning, confidence
    
    def _fallback_heuristic_relationship_reasoning(self, query: str) -> str:
        """Enhanced fallback heuristic reasoning for relationships"""
        
        logger.info("using_enhanced_heuristic_relationship_reasoning")
        
        query_lower = query.lower()
        words = query.split()
        word_count = len(words)
        
        # Enhanced relationship pattern detection
        breadth_indicators = [
            'compare', 'different', 'various', 'multiple', 'all types',
            'categories', 'classification', 'overview', 'broad'
        ]
        
        depth_indicators = [
            'cause', 'why', 'how', 'mechanism', 'pathway', 'process',
            'detailed', 'specific', 'deep', 'underlying', 'leads to'
        ]
        
        targeted_indicators = [
            'relationship', 'connection', 'association', 'linked',
            'between', 'and', 'versus', 'vs', 'correlation'
        ]
        
        # Score each strategy
        breadth_score = sum(1 for indicator in breadth_indicators if indicator in query_lower)
        depth_score = sum(1 for indicator in depth_indicators if indicator in query_lower)
        targeted_score = sum(1 for indicator in targeted_indicators if indicator in query_lower)
        
        # Consider query complexity
        if word_count > 15:
            breadth_score += 1  # Complex queries often need broad exploration
        elif word_count < 6:
            targeted_score += 1  # Simple queries often have specific focus
        
        # Consider learned performance as tiebreaker
        best_learned = self._get_best_learned_relationship_strategy()
        
        # Decision logic
        if breadth_score > max(depth_score, targeted_score):
            strategy = 'breadth_first'
        elif depth_score > max(breadth_score, targeted_score):
            strategy = 'depth_first'
        elif targeted_score > 0 or best_learned == 'targeted':
            strategy = 'targeted'
        elif best_learned:
            strategy = best_learned
        else:
            strategy = 'targeted'  # Safe default
        
        logger.info("enhanced_heuristic_relationship_reasoning_completed",
                   strategy=strategy,
                   breadth_score=breadth_score,
                   depth_score=depth_score,
                   targeted_score=targeted_score,
                   word_count=word_count)
        
        return strategy
    
    def _get_best_learned_relationship_strategy(self) -> Optional[str]:
        """Get best performing relationship strategy from learning"""
        
        if not any(perf['attempts'] > 0 for perf in self.search_strategies.values()):
            return None
        
        best_strategy = None
        best_score = 0.0
        
        for strategy, performance in self.search_strategies.items():
            if performance['attempts'] > 2:  # Only consider strategies with some history
                if performance['success_rate'] > best_score:
                    best_score = performance['success_rate']
                    best_strategy = strategy
        
        return best_strategy if best_score > 0.7 else None  # Only use if clearly better
        """AGENTIC REASONING: Analyze query to determine optimal relationship search strategy"""
        
        logger.info("reasoning_about_relationships", query_length=len(query))
        
        query_lower = query.lower()
        
        # Identify relationship patterns
        comparison_words = ['compare', 'versus', 'vs', 'difference', 'similar', 'contrast']
        causal_words = ['cause', 'effect', 'lead to', 'result in', 'because', 'due to']
        hierarchical_words = ['part of', 'contain', 'include', 'within', 'under', 'above']
        temporal_words = ['before', 'after', 'during', 'timeline', 'sequence', 'follow']
        
        relationship_indicators = {
            'comparison': any(word in query_lower for word in comparison_words),
            'causal': any(word in query_lower for word in causal_words),
            'hierarchical': any(word in query_lower for word in hierarchical_words),
            'temporal': any(word in query_lower for word in temporal_words)
        }
        
        # Determine strategy based on relationship patterns
        strategy_reasoning = []
        
        if relationship_indicators['comparison']:
            strategy = 'breadth_first'
            strategy_reasoning.append("Comparison query - need broad relationship exploration")
        elif relationship_indicators['causal']:
            strategy = 'depth_first'
            strategy_reasoning.append("Causal query - need deep chain exploration")
        elif relationship_indicators['hierarchical']:
            strategy = 'targeted'
            strategy_reasoning.append("Hierarchical query - need targeted structure exploration")
        elif relationship_indicators['temporal']:
            strategy = 'depth_first'
            strategy_reasoning.append("Temporal query - need sequential relationship exploration")
        else:
            # Use learned best strategy
            best_strategy = self._get_best_graph_strategy()
            if best_strategy:
                strategy = best_strategy
                strategy_reasoning.append(f"Using learned best strategy: {best_strategy}")
            else:
                strategy = 'breadth_first'
                strategy_reasoning.append("Default strategy - broad exploration")
        
        reasoning = "; ".join(strategy_reasoning)
        
        logger.info("relationship_strategy_reasoned",
                   strategy=strategy,
                   reasoning=reasoning,
                   relationship_indicators=relationship_indicators)
        
        return strategy
    
    def _get_best_graph_strategy(self) -> Optional[str]:
        """Get the best performing graph search strategy from learning"""
        
        if not any(perf['attempts'] > 0 for perf in self.search_strategies.values()):
            return None
        
        best_strategy = None
        best_score = 0.0
        
        for strategy, performance in self.search_strategies.items():
            if performance['attempts'] > 2:  # Need some history
                # Weight success rate heavily, with depth efficiency bonus
                efficiency_bonus = 1.0 / max(1, performance['avg_depth']) * 0.1
                combined_score = performance['success_rate'] + efficiency_bonus
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_strategy = strategy
        
        logger.debug("best_graph_strategy_determined",
                    strategy=best_strategy,
                    score=best_score)
        
        return best_strategy
    
    def _optimize_graph_parameters(self, query: str, strategy: str):
        """AGENTIC OPTIMIZATION: Adapt graph search parameters"""
        
        original_params = self.adaptive_params.copy()
        
        query_complexity = len(query.split())
        
        # Strategy-specific optimizations
        if strategy == 'breadth_first':
            self.adaptive_params['max_breadth'] = min(15, max(8, query_complexity))
            self.adaptive_params['max_depth'] = 2
            self.adaptive_params['relationship_threshold'] = 0.6
        elif strategy == 'depth_first':
            self.adaptive_params['max_breadth'] = min(8, max(5, query_complexity // 2))
            self.adaptive_params['max_depth'] = min(5, max(3, query_complexity // 3))
            self.adaptive_params['relationship_threshold'] = 0.75
        elif strategy == 'targeted':
            self.adaptive_params['max_breadth'] = min(12, max(6, query_complexity))
            self.adaptive_params['max_depth'] = 3
            self.adaptive_params['relationship_threshold'] = 0.8
        
        # Learning-based fine-tuning
        if strategy in self.search_strategies:
            perf = self.search_strategies[strategy]
            if perf['attempts'] > 3:
                if perf['success_rate'] < 0.6:
                    # Expand search if strategy is failing
                    self.adaptive_params['max_breadth'] = int(self.adaptive_params['max_breadth'] * 1.3)
                    self.adaptive_params['relationship_threshold'] *= 0.9
                elif perf['avg_depth'] > 4:
                    # Reduce depth if searches are getting too deep
                    self.adaptive_params['max_depth'] = max(2, self.adaptive_params['max_depth'] - 1)
        
        self.adaptive_params['search_strategy'] = strategy
        self.query_optimizations += 1
        
        logger.info("graph_parameters_optimized",
                   strategy=strategy,
                   optimization_count=self.query_optimizations,
                   original_params=original_params,
                   optimized_params=self.adaptive_params)
    
    def _execute_reasoned_graph_search(self, query: str, strategy: str) -> GraphSearchResult:
        """Execute graph search with reasoned strategy"""
        
        logger.info("executing_reasoned_graph_search", strategy=strategy)
        
        try:
            # For now, return empty results as this is a demonstration
            # In a real implementation, this would execute actual graph queries
            # based on the strategy (breadth_first, depth_first, targeted)
            
            documents = []
            
            # Simulate strategy-specific behavior
            if strategy == 'breadth_first':
                # Would execute broad relationship exploration
                pass
            elif strategy == 'depth_first':
                # Would execute deep chain exploration
                pass
            elif strategy == 'targeted':
                # Would execute focused relationship discovery
                pass
            
            result = GraphSearchResult(
                documents=documents,
                total_found=len(documents),
                optimizations_applied=self.query_optimizations
            )
            
            logger.info("reasoned_graph_search_completed",
                       strategy=strategy,
                       documents_found=len(documents),
                       optimizations_applied=self.query_optimizations)
            
            return result
            
        except Exception as e:
            logger.error("reasoned_graph_search_failed", strategy=strategy, error=str(e))
            return GraphSearchResult(
                documents=[],
                total_found=0,
                optimizations_applied=self.query_optimizations
            )
    
    def _learn_from_graph_outcome(self, query: str, strategy: str, result: GraphSearchResult):
        """AGENTIC LEARNING: Learn from graph search outcomes"""
        
        logger.info("learning_from_graph_outcome",
                   strategy=strategy,
                   documents_found=result.total_found)
        
        # Update strategy performance
        if strategy in self.search_strategies:
            perf = self.search_strategies[strategy]
            perf['attempts'] += 1
            
            # Calculate success (got results)
            success = result.total_found > 0
            current_successes = perf['success_rate'] * (perf['attempts'] - 1)
            new_success_rate = (current_successes + (1.0 if success else 0.0)) / perf['attempts']
            perf['success_rate'] = new_success_rate
            
            # Update average depth (using optimizations as proxy for complexity/depth)
            current_depth_total = perf['avg_depth'] * (perf['attempts'] - 1)
            current_depth = max(1, result.optimizations_applied)  # Use optimizations as depth proxy
            perf['avg_depth'] = (current_depth_total + current_depth) / perf['attempts']
            
            logger.info("graph_strategy_performance_updated",
                       strategy=strategy,
                       attempts=perf['attempts'],
                       success_rate=perf['success_rate'],
                       avg_depth=perf['avg_depth'])
        
        # Learn relationship patterns for future queries
        query_pattern = self._extract_relationship_pattern(query)
        if query_pattern:
            if query_pattern not in self.relationship_patterns:
                self.relationship_patterns[query_pattern] = {
                    'best_strategy': strategy,
                    'success_count': 1 if result.total_found > 0 else 0,
                    'total_attempts': 1
                }
            else:
                pattern_info = self.relationship_patterns[query_pattern]
                pattern_info['total_attempts'] += 1
                if result.total_found > 0:
                    pattern_info['success_count'] += 1
                    # Update best strategy if this one is more successful
                    if (pattern_info['success_count'] / pattern_info['total_attempts']) > 0.7:
                        pattern_info['best_strategy'] = strategy
            
            logger.debug("relationship_pattern_learned",
                        pattern=query_pattern,
                        strategy=strategy,
                        pattern_info=self.relationship_patterns[query_pattern])
    
    def _extract_relationship_pattern(self, query: str) -> Optional[str]:
        """Extract relationship pattern from query for learning"""
        
        query_lower = query.lower()
        
        # Simple pattern extraction
        if any(word in query_lower for word in ['compare', 'versus', 'difference']):
            return 'comparison'
        elif any(word in query_lower for word in ['cause', 'effect', 'lead to']):
            return 'causal'
        elif any(word in query_lower for word in ['part of', 'contain', 'include']):
            return 'hierarchical'
        elif any(word in query_lower for word in ['before', 'after', 'timeline']):
            return 'temporal'
        else:
            return 'general'
    
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
        
        # Enhanced synthesis using secure LLM interaction
        try:
            # Use LLM to generate comprehensive answer with security
            logger.info("llm_synthesis_started", context_pieces=len(context_pieces))
            
            # Use secure LLM interaction with proper template
            validated_content = secure_llm_interaction(
                llm=self.llm,
                template=ENHANCED_SYNTHESIS_TEMPLATE,
                user_input=query,
                combined_context=combined_context
            )
            
            if validated_content:
                synthesized_answer = str(validated_content).strip()
                
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
            else:
                logger.warning("secure_llm_synthesis_failed", error="No validated content returned")
                synthesized_answer = "Unable to generate answer from provided context."
                confidence_score = 0.0
            
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
    
    def get_agentic_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights into agentic behavior across all agents"""
        
        logger.info("generating_agentic_insights")
        
        # Orchestrator insights
        orchestrator_stats = self.orchestrator.get_learning_stats()
        orchestrator_reasoning = len(self.orchestrator.reasoning_history)
        
        # Vector agent insights
        vector_insights = {
            "search_strategies": self.vector_agent.strategy_performance,
            "search_history_length": len(self.vector_agent.search_history),
            "adaptive_params": self.vector_agent.adaptive_params
        }
        
        # Graph agent insights
        graph_insights = {
            "relationship_patterns": len(self.graph_agent.relationship_patterns),
            "search_strategies": self.graph_agent.search_strategies,
            "optimization_count": self.graph_agent.query_optimizations,
            "adaptive_params": self.graph_agent.adaptive_params
        }
        
        insights = {
            "orchestrator": {
                "learning_stats": orchestrator_stats,
                "reasoning_history_length": orchestrator_reasoning,
                "available_tools": len(self.orchestrator.available_tools),
                "tool_performance": {
                    tool: info["success_rate"] 
                    for tool, info in self.orchestrator.available_tools.items()
                }
            },
            "vector_agent": vector_insights,
            "graph_agent": graph_insights,
            "workflow": {
                "total_executions": self.execution_count,
                "agentic_capabilities": {
                    "dynamic_reasoning": True,
                    "goal_oriented_planning": True,
                    "adaptive_learning": True,
                    "tool_orchestration": True,
                    "strategy_optimization": True
                }
            }
        }
        
        logger.info("agentic_insights_generated",
                   total_patterns=len(self.graph_agent.relationship_patterns),
                   reasoning_steps=orchestrator_reasoning,
                   tool_count=len(self.orchestrator.available_tools))
        
        return insights
    
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
