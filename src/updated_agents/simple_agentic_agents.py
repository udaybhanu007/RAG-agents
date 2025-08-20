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

CRITICAL INSTRUCTIONS FOR SYNTHESIS:
1. The context above ALWAYS contains valid data from our medical database - NEVER say "No data found"
2. Answer using the information from the provided context
3. If any context is provided, there IS data in our database - provide a comprehensive answer based on that data
4. You may use the phrase "Based on the provided information" or "According to the database context"

SPECIAL HANDLING FOR COUNT QUERIES:
- When you see "Total count: X" in the context, THIS IS THE ANSWER to count queries
- Individual examples (Source 2, Source 3, etc.) are samples, NOT the complete count
- Always prioritize the "Total count" value over counting individual examples
- For demographic queries asking "how many", report the total count clearly
- Example: If context shows "Total count: 604", answer should include "604" as the count

NEVER RESPOND WITH "No data found" IF ANY CONTEXT IS PROVIDED.

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
        best_route = 'both'  # Default route
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

class EntityExtraction(BaseModel):
    """Enhanced entity extraction with demographics support"""
    entities: List[str] = Field(default_factory=list, description="Medical entities (diseases, symptoms, treatments)")
    relationships: List[str] = Field(default_factory=list, description="Relationship words (has, causes, treats)")
    concepts: List[str] = Field(default_factory=list, description="Medical concepts/specialties")
    demographics: List[str] = Field(default_factory=list, description="Demographic attributes (age, gender, ethnicity)")
    scenario: str = Field(default="SINGLE_ENTITY_BASIC", description="Extraction scenario classification")

class GraphQueryResult(BaseModel):
    """Enhanced graph query result with execution details"""
    triples: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved graph triples")
    queries_executed: int = Field(default=0, description="Number of Cypher queries executed")
    scenario_used: str = Field(default="", description="Scenario used for query generation")
    total_found: int = Field(default=0, description="Total results found")

class GraphSearchResult(BaseModel):
    """Structured graph search result"""
    documents: List[Dict[str, Any]] = Field(description="Retrieved relationship data")
    total_found: int = Field(description="Total number of results found")
    optimizations_applied: int = Field(description="Number of optimizations applied")

def _determine_extraction_scenario(entities: List[str], relationships: List[str], 
                                  concepts: List[str], demographics: List[str]) -> str:
    """
    Determine the most appropriate extraction scenario based on extracted components.
    This guides the query generation strategy.
    """
    entity_count = len(entities)
    relationship_count = len(relationships)
    concept_count = len(concepts)
    demographic_count = len(demographics)
    
    # Complex scenarios with demographics
    if demographic_count > 0:
        if entity_count > 1 and relationship_count > 0:
            return "COMPLEX_MULTI_ENTITY_WITH_DEMOGRAPHICS"
        elif entity_count == 1 and (relationship_count > 0 or concept_count > 0):
            return "SINGLE_ENTITY_WITH_DEMOGRAPHICS"
        elif entity_count > 1:
            return "MULTI_ENTITY_WITH_DEMOGRAPHICS"
        elif concept_count > 0:
            return "CONCEPTS_WITH_DEMOGRAPHICS"
        else:
            return "SINGLE_ENTITY_WITH_DEMOGRAPHICS"
    
    # No demographics scenarios
    if entity_count > 1 and relationship_count > 0:
        return "MULTI_ENTITY_WITH_RELATIONSHIPS"
    elif entity_count == 1 and relationship_count > 0:
        return "SINGLE_ENTITY_WITH_RELATIONSHIPS"
    elif entity_count > 1:
        return "MULTI_ENTITY_BASIC"
    elif entity_count == 1:
        if concept_count > 0:
            return "SINGLE_ENTITY_COMPLEX"
        else:
            return "SINGLE_ENTITY_BASIC"
    elif concept_count > 0:
        return "CONCEPTS_ONLY"
    else:
        return "GENERAL_QUERY"

def _create_dynamic_triple_from_record(record, query_info: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    """
    Create a dynamic triple from a Neo4j record with enhanced metadata.
    This function adapts the triple structure based on the record content.
    """
    record_dict = dict(record)
    
    # Create base triple structure
    triple = {
        "subject": "",
        "predicate": "related_to",
        "object": "",
        "content": f"Query result for: {original_query}",
        "source": "neo4j_dynamic",
        "query_type": query_info.get('description', 'Dynamic query'),
        "record_type": "graph_relationship"
    }
    
    # Extract subject and object from record
    if record_dict:
        keys = list(record_dict.keys())
        if len(keys) >= 2:
            # Use first two elements as subject and object
            triple["subject"] = str(record_dict[keys[0]])
            triple["object"] = str(record_dict[keys[1]])
        elif len(keys) == 1:
            # Single element becomes both subject and object
            triple["subject"] = str(record_dict[keys[0]])
            triple["object"] = str(record_dict[keys[0]])
    
    # Add all record fields as additional metadata
    for key, value in record_dict.items():
        if key not in ["subject", "predicate", "object", "content"]:
            triple[key] = value
    
    return triple

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
                "function": self._validate_medical_relevance_with_llm,
                "description": "Validate if query is medical/healthcare related using LLM",
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
    def _validate_medical_relevance_with_llm(self, query: str) -> Dict[str, Any]:
        """LLM-driven medical relevance validation - ONLY called by Orchestrator"""
        logger.info("orchestrator_medical_validation_called", query_length=len(query))
        
        # Use LLM-driven validation with the orchestrator's LLM
        result = validate_medical_relevance(query, llm=self.llm)
        
        # Update tool usage statistics for agentic learning
        self.available_tools["medical_validation"]["usage_count"] += 1
        
        logger.debug("orchestrator_medical_validation_completed", 
                    is_medical=result.get('is_medical', False),
                    validation_method=result.get('validation_method', 'unknown'))
        
        return result

    @traceable(**get_traceable_config("AgenticOrchestratorAgent"))
    def validate_medical_relevance_tool(self, query: str) -> Dict[str, Any]:
        """Tool to validate medical relevance of a query - delegates to LLM-driven method"""
        logger.info("medical_relevance_tool_called", query_length=len(query))
        result = self._validate_medical_relevance_with_llm(query)
        logger.debug("medical_relevance_tool_completed", is_medical=result.get('is_medical', False))
        return result
    
    @traceable(**get_traceable_config("AgenticOrchestratorAgent"))
    def analyze_query_characteristics_tool(self, query: str) -> QueryAnalysis:
        """Tool to analyze query characteristics and return structured output"""
        logger.info("query_characteristics_tool_called", query_length=len(query))
        result = analyze_query_characteristics(query, self.llm)
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
        """Handle non-medical queries using orchestrator's validation result"""
        logger.info("handling_non_medical_query")
        medical_validation = state.get("medical_validation", {})
        state["final_answer"] = medical_validation.get("quick_response", 
            "I can only help with medical and healthcare-related questions.")
        state["sources"] = []
        state["confidence_score"] = 0.0
        logger.debug("non_medical_response_generated", 
                    validation_method=medical_validation.get("validation_method", "unknown"))
        return state
    
    def _analyze_query_with_reasoning(self, query: str) -> Dict[str, Any]:
        """Analyze query with simple reasoning using structured outputs"""
        
        logger.info("query_reasoning_analysis_started", query_length=len(query))
        
        # Use LLM-driven medical validation through the orchestrator's method
        validation_result = self._validate_medical_relevance_with_llm(query)
        
        # Use functions for query analysis  
        query_analysis = analyze_query_characteristics(query, self.llm)
        
        result = {
            'query': query,  # Add the actual query text
            'query_type': query_analysis.intent,
            'is_medical': validation_result.get('is_medical', False),
            'complexity': query_analysis.complexity,
            'entity_count': query_analysis.entity_count,
            'has_relationships': query_analysis.has_relationships,
            'medical_validation': validation_result  # Include full validation result
        }
        
        logger.info("query_reasoning_analysis_completed", 
                   query_type=result['query_type'],
                   is_medical=result['is_medical'],
                   complexity=result['complexity'])
        
        return result
    
    def _detect_graph_database_patterns(self, query: str) -> Dict[str, Any]:
        """
        LLM-powered detection of graph database query patterns with intelligent analysis.
        Uses LLM to understand context and determine optimal routing strategy.
        """
        logger.info("llm_graph_pattern_detection_started", query_length=len(query))
        
        try:
            # LLM-based graph pattern detection template
            graph_detection_template = """
You are an expert in medical database routing and query analysis. Analyze the following medical query to determine if it should use a graph database (Neo4j) or vector database.

<USER_QUERY>{user_query}</USER_QUERY>

GRAPH DATABASE INDICATORS:
- Demographics: age, gender, patient characteristics (age=17, male patients, female, elderly)
- Structured data: "total number", "count of", "how many", exact values, aggregations
- Relationships: connections between entities, medical relationships
- Exact matching: "equals", "=", specific criteria matching
- Patient-specific queries: individual patient data, medical records
- Finding labels: specific medical findings, diagnoses

VECTOR DATABASE INDICATORS:
- Conceptual searches: general medical concepts, semantic similarity
- Document retrieval: research papers, medical literature
- Broad medical topics: overviews, general information

ANALYSIS CRITERIA:
1. Does the query require structured patient data with specific demographic criteria?
2. Does it ask for exact matches, counts, or aggregations?
3. Does it involve relationships between medical entities?
4. Does it require precise filtering by patient characteristics?

Respond in this EXACT format:
RECOMMENDATION: [GRAPH|VECTOR|HYBRID]
CONFIDENCE: [HIGH|MEDIUM|LOW]
REASONING: [Brief explanation of why this database type is optimal]
FORCE_GRAPH: [YES|NO] (YES if query absolutely requires graph database)
PATTERNS: [List 3-5 specific patterns detected in the query]
SCORE: [0.0-1.0] (How strongly this indicates graph database usage)
"""
            
            # Use secure LLM interaction for pattern detection
            validated_response = secure_llm_interaction(
                llm=self.llm,
                template=graph_detection_template,
                user_input=query
            )
            
            # Parse LLM response
            result = self._parse_graph_detection_response(validated_response, query)
            
            logger.info("llm_graph_pattern_detection_completed",
                       recommendation=result.get('recommendation', 'UNKNOWN'),
                       confidence=result.get('confidence', 'UNKNOWN'),
                       force_graph=result.get('force_graph', False),
                       score=result.get('score', 0.0))
            
            return result
            
        except Exception as e:
            logger.error("llm_graph_pattern_detection_failed", error=str(e))
            # Return default result without fallback
            return {
                'recommendation': 'VECTOR',
                'confidence': 'LOW',
                'reasoning': 'LLM detection failed',
                'force_graph': False,
                'patterns': [],
                'score': 0.0,
                'reasons': []
            }
    
    def _parse_graph_detection_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """Parse LLM response for graph database pattern detection"""
        result = {
            'recommendation': 'VECTOR',
            'confidence': 'MEDIUM',
            'reasoning': 'LLM analysis for optimal database routing',
            'force_graph': False,
            'patterns': [],
            'score': 0.5,
            'reasons': []
        }
        
        try:
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('RECOMMENDATION:'):
                    rec = line.split(':', 1)[1].strip().upper()
                    if rec in ['GRAPH', 'VECTOR', 'HYBRID']:
                        result['recommendation'] = rec
                        
                elif line.startswith('CONFIDENCE:'):
                    conf = line.split(':', 1)[1].strip().upper()
                    if conf in ['HIGH', 'MEDIUM', 'LOW']:
                        result['confidence'] = conf
                        
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()
                    if reasoning and len(reasoning) < 200:
                        result['reasoning'] = reasoning
                        result['reasons'].append(reasoning)
                        
                elif line.startswith('FORCE_GRAPH:'):
                    force = line.split(':', 1)[1].strip().upper()
                    result['force_graph'] = (force == 'YES')
                    
                elif line.startswith('PATTERNS:'):
                    patterns_text = line.split(':', 1)[1].strip()
                    if patterns_text:
                        # Extract patterns from text
                        patterns = [p.strip() for p in patterns_text.split(',') if p.strip()]
                        result['patterns'] = patterns[:5]  # Limit to 5 patterns
                        
                elif line.startswith('SCORE:'):
                    try:
                        score_text = line.split(':', 1)[1].strip()
                        score = float(score_text)
                        result['score'] = max(0.0, min(1.0, score))  # Clamp to 0-1
                    except ValueError:
                        result['score'] = 0.5  # Default if parsing fails
            
            # Adjust score based on recommendation
            if result['recommendation'] == 'GRAPH':
                result['score'] = max(0.7, result['score'])  # Boost for graph recommendation
            elif result['recommendation'] == 'VECTOR':
                result['score'] = min(0.3, result['score'])  # Lower for vector recommendation
            elif result['recommendation'] == 'HYBRID':
                result['score'] = 0.6  # Balanced score for hybrid
            
            logger.debug("graph_detection_response_parsed",
                        recommendation=result['recommendation'],
                        confidence=result['confidence'],
                        score=result['score'],
                        patterns_count=len(result['patterns']))
            
        except Exception as e:
            logger.error("graph_detection_response_parsing_failed", error=str(e))
            # Return default result without fallback
            result = {
                'recommendation': 'VECTOR',
                'confidence': 'LOW',
                'reasoning': 'Response parsing failed',
                'force_graph': False,
                'patterns': [],
                'score': 0.0,
                'reasons': []
            }
        
        return result
    
    def _make_enhanced_agentic_routing_decision(self, analysis: Dict[str, Any], execution_goals: List[Dict]) -> SimpleReasoningPlan:
        """ENHANCED AGENTIC BEHAVIOR: Goal-aware routing with dynamic tool selection"""
        
        query_type = analysis['query_type']
        complexity = analysis.get('complexity', 'simple')
        query = analysis.get('query', '')  # Get the actual query text
        
        logger.info("making_enhanced_agentic_routing_decision", 
                   query_type=query_type,
                   complexity=complexity,
                   goal_count=len(execution_goals))
        
        # ENHANCED GRAPH DATABASE DETECTION (LLM-POWERED)
        graph_indicators = self._detect_graph_database_patterns(query)
        
        # ADVANCED LEARNING APPLICATION: Consider tool performance history
        if learning_memory.routing_performance and not graph_indicators.get('force_graph', False):
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
        
        # ENHANCED REASONING-BASED ROUTING: LLM recommendation takes priority
        reasoning_factors = []
        
        # PRIORITY 1: LLM-based graph database recommendation (overrides other logic)
        llm_recommendation = graph_indicators.get('recommendation', 'VECTOR').upper()
        llm_confidence = graph_indicators.get('confidence', 'MEDIUM').upper()
        
        if graph_indicators.get('force_graph', False) or llm_recommendation == 'GRAPH':
            route = 'graph'
            reasoning_factors.append(f"LLM analysis recommends GRAPH database ({llm_confidence} confidence)")
            reasoning_factors.append(graph_indicators.get('reasoning', 'LLM-based recommendation'))
            if graph_indicators.get('patterns'):
                reasoning_factors.append(f"Detected patterns: {', '.join(graph_indicators['patterns'][:3])}")
            logger.info("llm_graph_recommendation_applied", 
                       recommendation=llm_recommendation,
                       confidence=llm_confidence,
                       patterns=graph_indicators.get('patterns', []))
            
            # EARLY RETURN: LLM recommendation overrides all other logic
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
            
        elif llm_recommendation == 'HYBRID':
            route = 'both'
            reasoning_factors.append(f"LLM analysis recommends HYBRID approach ({llm_confidence} confidence)")
            reasoning_factors.append(graph_indicators.get('reasoning', 'LLM-based hybrid recommendation'))
            
            # EARLY RETURN: LLM hybrid recommendation overrides other logic
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
            
        elif graph_indicators.get('score', 0.0) > 0.6:
            route = 'graph'
            reasoning_factors.append(f"High graph pattern score ({graph_indicators.get('score', 0.0):.2f})")
            reasoning_factors.extend(graph_indicators.get('reasons', []))
            
            # EARLY RETURN: High graph score overrides traditional logic
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
        
        # Traditional complexity-based routing
        elif complexity == 'complex':
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
                route = 'both'
        
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
        
        # DYNAMIC GOAL ADAPTATION: Update search tool based on routing decision
        primary_goal = goals[0] if goals else {}
        sub_goals = primary_goal.get("sub_goals", [])
        
        # Update the search sub-goal to match the routing decision
        for sub_goal in sub_goals:
            if sub_goal.get("id") == "search_data":
                original_tool = sub_goal.get("tool", "")
                if plan.selected_route == "graph":
                    sub_goal["tool"] = "graph_search"
                    sub_goal["description"] = "Search graph database for structured information"
                elif plan.selected_route == "both":
                    sub_goal["tool"] = "hybrid_search"
                    sub_goal["description"] = "Search both vector and graph databases"
                elif plan.selected_route == "vector":
                    sub_goal["tool"] = "vector_search"
                    sub_goal["description"] = "Search vector database for semantic information"
                
                if sub_goal["tool"] != original_tool:
                    logger.info("sub_goal_tool_updated", 
                               original_tool=original_tool,
                               new_tool=sub_goal["tool"],
                               routing_decision=plan.selected_route)
        
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
        
        # ENHANCED LLM-POWERED ROUTING: Use LLM graph detection
        logger.info("applying_enhanced_llm_powered_routing", query_type=query_type)
        
        # Get LLM graph detection recommendation
        graph_detection = self._detect_graph_database_patterns(query)
        
        if graph_detection.get('force_graph', False):
            route = 'graph'
            reasoning = f"LLM detected graph patterns: {graph_detection.get('patterns', [])} (confidence: {graph_detection.get('confidence', 'UNKNOWN')})"
        elif query_type == 'comparison':
            route = 'both'  # Comparisons benefit from both vector and graph
            reasoning = "Complex comparisons require both semantic similarity and relationship analysis"
        elif query_type == 'relational':
            route = 'graph'  # Relationships are graph strengths
            reasoning = "Relationship queries are optimally handled by graph database"
        elif query_type == 'analytical':
            route = 'both'  # Analysis benefits from comprehensive data
            reasoning = "Analytical queries need comprehensive data from both sources"
        else:  # factual - check if graph patterns detected
            if graph_detection.get('recommendation') == 'GRAPH':
                route = 'graph'
                reasoning = f"LLM recommended graph search for factual query (confidence: {graph_detection.get('confidence', 'UNKNOWN')})"
            else:
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
            # Handle non-medical queries
            logger.debug("executing_non_medical_handling")
            return self.handle_non_medical_query(state)
    
    def _execute_vector_search(self, state: WorkflowState) -> WorkflowState:
        """Execute vector search route using real AgenticVectorRAGAgent"""
        logger.info("executing_vector_search_route")
        
        # Check medical validation already performed by orchestrator
        medical_validation = state.get("medical_validation", {})
        if not medical_validation.get("is_medical", False):
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
        
        # Check medical validation already performed by orchestrator
        medical_validation = state.get("medical_validation", {})
        if not medical_validation.get("is_medical", False):
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
        
        # Check medical validation already performed by orchestrator
        medical_validation = state.get("medical_validation", {})
        if not medical_validation.get("is_medical", False):
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
            # Return default strategy without fallback
            return "hybrid"
    
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
        
        strategy = "hybrid"  # Default strategy
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
    Simplified Graph Agent with targeted strategy and tool-based approach
    
    Key Features:
    1. Entity extraction using comprehensive prompt templates
    2. Cypher query generation using LLM-driven tools
    3. Targeted graph search strategy only
    4. Tool-based architecture for entity extraction and cypher generation
    """
    
    def __init__(self, llm: AzureChatOpenAI, graph_store):
        super().__init__(AgentRole.GRAPH_RAG)
        self.llm = llm
        
        # Handle different graph_store formats
        if isinstance(graph_store, dict) and graph_store.get('type') == 'neo4j':
            # Create actual Neo4j driver from connection details
            from neo4j import GraphDatabase
            self.graph_store = GraphDatabase.driver(
                graph_store['uri'], 
                auth=(graph_store['username'], graph_store['password'])
            )
            logger.info("created_neo4j_driver_from_config", 
                       uri=graph_store['uri'], 
                       username=graph_store['username'])
        else:
            # Use the provided graph_store as-is (should be a Neo4j driver)
            self.graph_store = graph_store
            logger.info("using_provided_graph_store", 
                       graph_store_type=type(graph_store).__name__)
        
        # Simple tracking for query execution
        self.query_count = 0
        
        logger.info("simplified_graph_agent_initialized", 
                   strategy="targeted_only")
    
    @traceable(**get_traceable_config("AgenticGraphRAGAgent"))
    def search_with_optimization(self, state: WorkflowState) -> WorkflowState:
        """Simplified graph search using targeted strategy and tools"""
        
        logger.info("simplified_graph_search_started")
        
        query = state["query"]
        
        # Execute targeted graph search using tools
        search_result = self._execute_targeted_graph_search(query)
        
        # Add structured result to state
        state["graph_results"] = search_result.dict()
        
        logger.info("simplified_graph_search_completed",
                   documents_found=search_result.total_found,
                   query_count=self.query_count)
        return state
    
    @traceable(**get_traceable_config("AgenticGraphRAGAgent"))
    def extract_entities_and_relationships_tool(self, query: str) -> EntityExtraction:
        """Tool for extracting entities and relationships using comprehensive prompt template"""
        logger.info("entity_extraction_tool_started", query_length=len(query))
        
        # Import here to avoid circular imports
        from core.input_sanitization import (
            detect_prompt_injection,
            secure_llm_interaction
        )
        
        # Step 1: Detect prompt injection attempts
        if detect_prompt_injection(query):
            logger.warning("prompt_injection_blocked_in_extraction_tool", query_snippet=query[:50])
            return EntityExtraction(
                entities=[],
                relationships=[],
                concepts=[],
                demographics=[],
                scenario="PROMPT_INJECTION_DETECTED"
            )
        
        # Step 2: Use comprehensive entity extraction template
        comprehensive_template = """
You are a medical knowledge graph entity extraction expert. Analyze the following medical query and extract ALL relevant information in the specified format.

<USER_QUERY>{user_query}</USER_QUERY>

Extract the following information:

1. MEDICAL ENTITIES: Diseases, symptoms, treatments, medications, procedures, body parts, medical conditions
   **SPECIAL HANDLING FOR COMPOUND CONDITIONS**: If query mentions multiple conditions connected by "and" 
   (e.g., "Consolidation and Effusion"), extract each condition as a SEPARATE entity, not as one compound entity.
   
2. RELATIONSHIPS: Connection words (has, causes, treats, affects, diagnosed_with, prescribed_for, related_to)
   **FOR COMPOUND CONDITIONS**: When multiple conditions are mentioned with "and", use relationship "has_multiple"
   
3. MEDICAL CONCEPTS: Medical domains/specialties (cardiology, oncology, neurology, pediatrics, etc.)
4. DEMOGRAPHICS: Age, gender, ethnicity, location, patient characteristics

CRITICAL RULES FOR COMPOUND CONDITIONS:
- "Consolidation and Effusion" -> ENTITIES: [Consolidation, Effusion] (separate, not compound)
- "Pneumonia and Cardiomegaly" -> ENTITIES: [Pneumonia, Cardiomegaly] (separate entities)
- This enables proper Cypher generation for patients having BOTH conditions

Format your response EXACTLY as follows:
ENTITIES: [entity1, entity2, entity3]
RELATIONSHIPS: [relationship1, relationship2]
CONCEPTS: [concept1, concept2]
DEMOGRAPHICS: [demographic1, demographic2]

Be comprehensive and include ALL relevant medical terms, even if they seem obvious.
If a category has no items, use empty brackets: []

Examples:
- "chest pain in elderly women" -> ENTITIES: [chest pain], RELATIONSHIPS: [has], CONCEPTS: [cardiology], DEMOGRAPHICS: [elderly, women]
- "diabetes treatment options" -> ENTITIES: [diabetes, treatment], RELATIONSHIPS: [treats], CONCEPTS: [endocrinology], DEMOGRAPHICS: []
- "Consolidation and Effusion in patients" -> ENTITIES: [Consolidation, Effusion, patients], RELATIONSHIPS: [has_multiple], CONCEPTS: [pulmonology], DEMOGRAPHICS: []
- "Pneumonia and Cardiomegaly" -> ENTITIES: [Pneumonia, Cardiomegaly], RELATIONSHIPS: [has_multiple], CONCEPTS: [pulmonology, cardiology], DEMOGRAPHICS: []
"""
        
        try:
            # Use secure LLM interaction
            response = secure_llm_interaction(
                llm=self.llm,
                template=comprehensive_template,
                user_input=query
            )
            
            # Parse the response
            entities = []
            relationships = []
            concepts = []
            demographics = []
            
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('ENTITIES:'):
                    entities_str = line.split(':', 1)[1].strip()
                    entities = self._parse_list_from_string(entities_str)
                elif line.startswith('RELATIONSHIPS:'):
                    relationships_str = line.split(':', 1)[1].strip()
                    relationships = self._parse_list_from_string(relationships_str)
                elif line.startswith('CONCEPTS:'):
                    concepts_str = line.split(':', 1)[1].strip()
                    concepts = self._parse_list_from_string(concepts_str)
                elif line.startswith('DEMOGRAPHICS:'):
                    demographics_str = line.split(':', 1)[1].strip()
                    demographics = self._parse_list_from_string(demographics_str)
            
            # Determine scenario based on extracted content
            scenario = "GENERAL_QUERY"
            if demographics:
                scenario = "DEMOGRAPHIC_QUERY"
            elif any(concept in ['cardiology', 'oncology', 'neurology'] for concept in concepts):
                scenario = "SPECIALIZED_QUERY"
            
            result = EntityExtraction(
                entities=entities,
                relationships=relationships,
                concepts=concepts,
                demographics=demographics,
                scenario=scenario
            )
            
            logger.info("entity_extraction_tool_completed",
                       entities_count=len(entities),
                       relationships_count=len(relationships),
                       concepts_count=len(concepts),
                       demographics_count=len(demographics),
                       scenario=scenario)
            
            return result
            
        except Exception as e:
            logger.error("entity_extraction_tool_failed", error=str(e))
            return EntityExtraction(
                entities=[],
                relationships=[],
                concepts=[],
                demographics=[],
                scenario="EXTRACTION_ERROR"
            )
    
    @traceable(**get_traceable_config("AgenticGraphRAGAgent"))
    def generate_cypher_query_tool(self, extraction: EntityExtraction, original_query: str) -> List[Dict[str, Any]]:
        """Tool for generating Cypher queries using comprehensive prompt template"""
        logger.info("cypher_generation_tool_started", 
                   scenario=extraction.scenario, 
                   entities_count=len(extraction.entities))
        
        # Import here to avoid circular imports
        from core.input_sanitization import secure_llm_interaction
        
        try:
            # Comprehensive Cypher generation template
            cypher_template = """
You are a Neo4j Cypher query expert for medical databases. Generate optimized Cypher queries based on the extracted entities and demographics.

ORIGINAL QUERY: {original_query}
EXTRACTED ENTITIES: {entities}
DEMOGRAPHICS: {demographics}
MEDICAL CONCEPTS: {concepts}
RELATIONSHIPS: {relationships}

IMPORTANT DATABASE SCHEMA (USE EXACT PROPERTY NAMES):
- Patient nodes: properties are gender ('M'/'F'), age (integer), patient_id, id
- Finding nodes: properties are finding_label (NOT label), name, occurrence_count
- Relationships: (Patient)-[:HAS_FINDING]->(Finding)

CRITICAL RULES:
1. Use f.finding_label (NOT f.label) for finding labels
2. Use CONTAINS for partial matching of finding labels
3. Use COUNT(DISTINCT p) for patient counts to avoid duplicates
4. Always use exact property names from schema above
5. **COMPOUND FINDINGS**: If query mentions multiple findings connected by "and" (e.g., "Consolidation and Effusion"), 
   generate queries for patients having BOTH conditions separately, not as a single compound label
6. **MULTIPLE CONDITIONS**: Use separate MATCH patterns for each condition when "and" appears between medical terms

SPECIAL HANDLING FOR COMPOUND FINDINGS:
- Query: "patients with Consolidation and Effusion" should generate:
  MATCH (p:Patient)-[:HAS_FINDING]->(f1:Finding), (p)-[:HAS_FINDING]->(f2:Finding)
  WHERE toLower(f1.finding_label) CONTAINS 'consolidation' 
  AND toLower(f2.finding_label) CONTAINS 'effusion'
  RETURN COUNT(DISTINCT p) as total_count

- NOT: WHERE f.finding_label = 'Consolidation and Effusion' (exact match)
- NOT: WHERE f.finding_label CONTAINS 'consolidation and effusion' (single label)

Generate 1-3 Cypher queries that would answer the original question. Focus on:
1. Demographic filtering (age, gender)
2. Finding/condition matching using finding_label
3. Count aggregations when appropriate

Format each query as:
QUERY_TYPE: [descriptive name]
CYPHER: [cypher query]
DESCRIPTION: [what this query does]

CRITICAL RULES FOR DYNAMIC GENERATION:
1. Base queries on the EXTRACTED data above (entities, demographics, concepts)
2. Use f.finding_label (NOT f.label) for finding labels
3. Use CONTAINS for partial matching: toLower(f.finding_label) CONTAINS 'term'
4. Use COUNT(DISTINCT p) for patient counts to avoid duplicates
5. Include demographic filters based on extracted demographics
6. Generate contextually appropriate queries for the specific scenario
7. **HANDLE COMPOUND FINDINGS**: When entities contain multiple medical conditions connected by "and",
   generate queries that find patients having ALL conditions separately, not as single compound labels
8. **MULTIPLE FINDING PATTERN**: For "X and Y" findings, use multiple MATCH patterns:
   MATCH (p:Patient)-[:HAS_FINDING]->(f1:Finding), (p)-[:HAS_FINDING]->(f2:Finding)
   WHERE condition1 AND condition2
"""
            
            # Format the template with extracted data
            formatted_template = cypher_template.format(
                original_query=original_query,
                entities=extraction.entities,
                demographics=extraction.demographics,
                concepts=extraction.concepts,
                relationships=extraction.relationships
            )
            
            # Use secure LLM interaction
            response = secure_llm_interaction(
                llm=self.llm,
                template=formatted_template,
                user_input=""
            )
            
            # Parse the response to extract queries - handle markdown code blocks
            queries = []
            current_query = {}
            in_cypher_block = False
            cypher_lines = []
            
            lines = response.split('\n')
            for line in lines:
                line_stripped = line.strip()
                
                # Check for markdown headers (### QUERY_TYPE:) or regular QUERY_TYPE:
                if line_stripped.startswith('###') and 'QUERY_TYPE:' in line_stripped:
                    if current_query:
                        queries.append(current_query)
                    current_query = {'query_type': line_stripped.split('QUERY_TYPE:', 1)[1].strip()}
                elif line_stripped.startswith('QUERY_TYPE:'):
                    if current_query:
                        queries.append(current_query)
                    current_query = {'query_type': line_stripped.split(':', 1)[1].strip()}
                
                # Handle CYPHER section with potential code blocks
                elif line_stripped.startswith('CYPHER:'):
                    cypher_content = line_stripped.split(':', 1)[1].strip()
                    if cypher_content:  # Cypher on same line
                        current_query['cypher'] = cypher_content
                    else:  # Cypher on following lines (potentially in code block)
                        in_cypher_block = True
                        cypher_lines = []
                
                # Handle code block markers
                elif in_cypher_block:
                    if line_stripped.startswith('```'):
                        if line_stripped == '```' or line_stripped == '```cypher':
                            # Start or end of code block, skip the marker
                            continue
                        else:
                            # End of code block
                            in_cypher_block = False
                            current_query['cypher'] = '\n'.join(cypher_lines).strip()
                            cypher_lines = []
                    else:
                        # Content inside code block
                        cypher_lines.append(line)
                
                # Handle DESCRIPTION section
                elif line_stripped.startswith('DESCRIPTION:'):
                    # If we were still in cypher block, close it
                    if in_cypher_block:
                        in_cypher_block = False
                        current_query['cypher'] = '\n'.join(cypher_lines).strip()
                        cypher_lines = []
                    current_query['description'] = line_stripped.split(':', 1)[1].strip()
            
            # Handle final cypher block if response ends without closing
            if in_cypher_block and cypher_lines:
                current_query['cypher'] = '\n'.join(cypher_lines).strip()
            
            # Add the last query
            if current_query:
                queries.append(current_query)
            
            # Post-process queries to fix common schema issues
            for query in queries:
                if 'cypher' in query:
                    # Fix property name issues
                    cypher = query['cypher']
                    original_cypher = cypher
                    
                    # CRITICAL: Clean up Cypher query - remove any DESCRIPTION text that got included
                    lines = cypher.split('\n')
                    clean_cypher_lines = []
                    for line in lines:
                        # Stop at any line that starts with DESCRIPTION or other non-Cypher content
                        if line.strip().startswith(('DESCRIPTION:', 'These queries', '###', 'Note:')):
                            break
                        clean_cypher_lines.append(line)
                    
                    cypher = '\n'.join(clean_cypher_lines).strip()
                    
                    # CRITICAL: Validate query completeness - must have RETURN clause
                    if not cypher.strip().upper().endswith(('RETURN', 'RETURN COUNT(DISTINCT p) AS TOTAL_COUNT', 'RETURN COUNT(P) AS TOTAL_COUNT')) and 'RETURN' not in cypher.upper():
                        logger.warning("incomplete_cypher_query_detected", 
                                     incomplete_query=cypher,
                                     skipping_incomplete_query=True)
                        # Skip incomplete queries instead of using hardcoded fallbacks
                        continue
                    
                    # ADDITIONAL: Check for MATCH-only queries (common LLM truncation)
                    if cypher.strip().upper().startswith('MATCH') and 'WHERE' not in cypher.upper() and 'RETURN' not in cypher.upper():
                        logger.warning("match_only_query_detected", 
                                     incomplete_query=cypher,
                                     skipping_incomplete_query=True)
                        # Skip incomplete queries instead of using hardcoded fallbacks
                        continue
                    
                    # Only apply fixes if they are needed to avoid over-fixing
                    # Check for exact wrong patterns and replace them carefully
                    
                    # Fix f.label (but not f.finding_label)
                    if 'f.label' in cypher and 'f.finding_label' not in cypher:
                        cypher = cypher.replace('f.label', 'f.finding_label')
                    
                    # Fix partial property names only if they're wrong
                    # Use word boundaries to avoid replacing parts of correct properties
                    import re
                    
                    # Replace f.find only if it's not part of f.finding_label
                    if re.search(r'\bf\.find\b', cypher):
                        cypher = re.sub(r'\bf\.find\b', 'f.finding_label', cypher)
                    
                    # Replace f.condition only if it exists
                    if re.search(r'\bf\.condition\b', cypher):
                        cypher = re.sub(r'\bf\.condition\b', 'f.finding_label', cypher)
                    
                    # Ensure COUNT(DISTINCT p) for patient counts
                    if 'COUNT(p)' in cypher and 'total_count' in cypher:
                        cypher = cypher.replace('COUNT(p)', 'COUNT(DISTINCT p)')
                    
                    query['cypher'] = cypher
                    
                    # Log the fix with full query details
                    if cypher != original_cypher:
                        logger.info("cypher_query_auto_fixed", 
                                   original_full=original_cypher,
                                   fixed_full=cypher,
                                   changes_made=True)
                    else:
                        logger.info("cypher_query_validated", 
                                   query_full=cypher,
                                   changes_made=False)
            
            # No default query - let LLM generate proper queries
            if not queries:
                logger.warning("no_valid_queries_generated", 
                             original_query=original_query)
                return []
            
            logger.info("cypher_generation_tool_completed", queries_count=len(queries))
            return queries
            
        except Exception as e:
            logger.error("cypher_generation_tool_failed", error=str(e))
            # No hardcoded fallback - return empty list to let system handle gracefully
            return []
    
    def _parse_list_from_string(self, list_str: str) -> List[str]:
        """Parse a list from string format like '[item1, item2, item3]'"""
        if not list_str or list_str.strip() == '[]':
            return []
        
        # Remove brackets and split by comma
        cleaned = list_str.strip()
        if cleaned.startswith('[') and cleaned.endswith(']'):
            cleaned = cleaned[1:-1]
        
        items = [item.strip().strip("'\"") for item in cleaned.split(',') if item.strip()]
        return [item for item in items if item]  # Remove empty items
    
    def _execute_targeted_graph_search(self, query: str) -> GraphSearchResult:
        """Execute targeted graph search using the new tool-based approach"""
        
        logger.info("executing_targeted_graph_search", received_query=query)
        
        # Debug the graph_store object
        logger.info("debugging_graph_store_object", 
                   graph_store_id=id(self.graph_store),
                   graph_store_type=type(self.graph_store).__name__,
                   graph_store_repr=repr(self.graph_store),
                   graph_store_value=str(self.graph_store) if len(str(self.graph_store)) < 200 else str(self.graph_store)[:200] + "...")
        
        try:
            self.query_count += 1
            
            # STEP 1: Extract entities and relationships using tool
            extraction = self.extract_entities_and_relationships_tool(query)
            
            # STEP 2: Generate Cypher queries using tool
            cypher_queries = self.generate_cypher_query_tool(extraction, query)
            
            # STEP 3: Execute the generated queries
            documents = []
            
            logger.info("preparing_to_execute_cypher_queries", 
                       query_count=len(cypher_queries),
                       has_session_method=hasattr(self.graph_store, 'session'),
                       graph_store_type=type(self.graph_store).__name__,
                       graph_store_exists=bool(self.graph_store),
                       graph_store_methods=[m for m in dir(self.graph_store) if not m.startswith('_')])
            
            if self.graph_store and hasattr(self.graph_store, 'session'):
                try:
                    logger.info("opening_neo4j_session")
                    with self.graph_store.session() as session:
                        logger.info("neo4j_session_opened_successfully")
                        for i, query_info in enumerate(cypher_queries):
                            try:
                                cypher_query = query_info.get("cypher", "")
                                description = query_info.get("description", "Generated query")
                                
                                logger.info("processing_query_info", 
                                           query_index=i,
                                           has_cypher=bool(cypher_query),
                                           query_type=query_info.get("query_type", "unknown"))
                                
                                if not cypher_query:
                                    logger.warning("skipping_empty_cypher_query", query_index=i)
                                    continue
                                
                                logger.info("executing_targeted_cypher_query", 
                                           query_index=i,
                                           description=description,
                                           query_preview=cypher_query[:100],
                                           full_query=cypher_query)
                                
                                result = session.run(cypher_query)
                                records = list(result)
                                
                                logger.info("cypher_query_execution_completed",
                                           query_index=i,
                                           cypher=cypher_query,
                                           records_found=len(records))
                                
                                if len(records) == 0:
                                    logger.warning("cypher_query_returned_no_records", 
                                                 query_index=i,
                                                 full_query=cypher_query)
                                
                                for j, record in enumerate(records):
                                    record_dict = dict(record)
                                    
                                    logger.info("processing_record", 
                                               query_index=i,
                                               record_index=j,
                                               record_keys=list(record_dict.keys()),
                                               record_data=record_dict)
                                    
                                    # Create content based on record type
                                    if "total_count" in record_dict:
                                        content = f"Total count: {record_dict['total_count']}"
                                        doc = {
                                            "content": content,
                                            "metadata": {
                                                "source": "neo4j_targeted",
                                                "total_count": record_dict["total_count"],
                                                "query_type": query_info.get("query_type", "unknown"),
                                                "description": description,
                                                "strategy": "targeted"
                                            }
                                        }
                                        documents.append(doc)
                                        logger.info("created_total_count_document", 
                                                   query_index=i,
                                                   total_count=record_dict["total_count"],
                                                   document_content=content)
                                    else:
                                        # Handle other record types
                                        content = f"Graph data: {', '.join([f'{k}: {v}' for k, v in record_dict.items()])}"
                                        doc = {
                                            "content": content,
                                            "metadata": {
                                                "source": "neo4j_targeted",
                                                "query_type": query_info.get("query_type", "unknown"),
                                                "description": description,
                                                "strategy": "targeted",
                                                **record_dict
                                            }
                                        }
                                        documents.append(doc)
                                
                            except Exception as query_error:
                                logger.warning("targeted_cypher_query_failed", 
                                             error=str(query_error),
                                             query_preview=cypher_query[:100])
                                continue
                                
                        logger.info("targeted_neo4j_queries_completed", 
                                   documents_found=len(documents),
                                   queries_executed=len(cypher_queries))
                        
                except Exception as e:
                    logger.warning("targeted_neo4j_execution_failed", error=str(e))
            
            # Return result with documents found (or empty if none)
            result = GraphSearchResult(
                documents=documents,
                total_found=len(documents),
                optimizations_applied=self.query_count
            )
            
            logger.info("targeted_graph_search_completed",
                       documents_found=len(documents),
                       query_count=self.query_count)
            
            return result
            
        except Exception as e:
            logger.error("targeted_graph_search_failed", error=str(e))
            return GraphSearchResult(
                documents=[],
                total_found=0,
                optimizations_applied=self.query_count
            )
    
    @traceable(**get_traceable_config("AgenticGraphRAGAgent"))
    def search_graph(self, state: WorkflowState) -> GraphSearchResult:
        """Graph search implementation using targeted strategy only"""
        logger.debug("graph_search_started", query_count=self.query_count)
        
        try:
            query = state.get("query", "")
            
            # Use the simplified targeted search
            result = self._execute_targeted_graph_search(query)
            
            logger.debug("graph_search_completed",
                        documents_found=result.total_found,
                        query_count=self.query_count)
            
            return result
        except Exception as e:
            logger.error("graph_search_failed", error=str(e))
            return GraphSearchResult(
                documents=[],
                total_found=0,
                optimizations_applied=self.query_count
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
        
        # DEBUG: Log the state structure for debugging
        logger.info("DEBUG_VALIDATION_STATE", 
                   graph_results_key_exists="graph_results" in state,
                   graph_results_type=type(state.get("graph_results", {})),
                   graph_results_keys=list(state.get("graph_results", {}).keys()) if isinstance(state.get("graph_results", {}), dict) else "not_dict",
                   graph_results_documents_count=len(graph_results),
                   vector_results_count=len(vector_results),
                   state_keys=list(state.keys()))
        
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

# ==================== ENHANCED GRAPH SEARCH TOOLS ====================

@tool
def extract_entities_and_relationships(query: str, llm) -> EntityExtraction:
    """
    Enhanced entity extraction with demographics support and comprehensive medical analysis.
    Extracts entities, relationships, concepts, and demographics from medical queries.
    """
    logger.info("enhanced_entity_extraction_started", query_length=len(query))
    
    # Step 1: Detect prompt injection attempts
    if detect_prompt_injection(query):
        logger.warning("prompt_injection_blocked_in_extraction", query_snippet=query[:50])
        return EntityExtraction(
            entities=[],
            relationships=[],
            concepts=[],
            demographics=[],
            scenario="PROMPT_INJECTION_DETECTED"
        )
    
    # Step 2: Use comprehensive entity extraction template
    comprehensive_template = """
You are a medical knowledge graph entity extraction expert. Analyze the following medical query and extract ALL relevant information in the specified format.

<USER_QUERY>{user_query}</USER_QUERY>

Extract the following information:

1. MEDICAL ENTITIES: Diseases, symptoms, treatments, medications, procedures, body parts, medical conditions
2. RELATIONSHIPS: Connection words (has, causes, treats, affects, diagnosed_with, prescribed_for, related_to)
3. MEDICAL CONCEPTS: Medical domains/specialties (cardiology, oncology, neurology, pediatrics, etc.)
4. DEMOGRAPHICS: Age, gender, ethnicity, location, patient characteristics

Format your response EXACTLY as follows:
ENTITIES: [entity1, entity2, entity3]
RELATIONSHIPS: [relationship1, relationship2]
CONCEPTS: [concept1, concept2]
DEMOGRAPHICS: [demographic1, demographic2]

Be comprehensive and include ALL relevant medical terms, even if they seem obvious.
If a category has no items, use empty brackets: []

Examples:
- "chest pain in elderly women" -> ENTITIES: [chest pain], RELATIONSHIPS: [has], CONCEPTS: [cardiology], DEMOGRAPHICS: [elderly, women]
- "diabetes treatment options" -> ENTITIES: [diabetes, treatment], RELATIONSHIPS: [treats], CONCEPTS: [endocrinology], DEMOGRAPHICS: []
"""
    
    try:
        validated_content = secure_llm_interaction(
            llm=llm,
            template=comprehensive_template,
            user_input=query
        )
        
        # Parse response with enhanced extraction
        entities, relationships, concepts, demographics = [], [], [], []
        
        lines = validated_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('ENTITIES:'):
                entities_text = line.split('ENTITIES:')[1].strip()
                if entities_text and entities_text != '[]':
                    entities = [e.strip().strip('"\'') for e in entities_text.strip('[]').split(',') if e.strip()]
            elif line.startswith('RELATIONSHIPS:'):
                rel_text = line.split('RELATIONSHIPS:')[1].strip()
                if rel_text and rel_text != '[]':
                    relationships = [r.strip().strip('"\'') for r in rel_text.strip('[]').split(',') if r.strip()]
            elif line.startswith('CONCEPTS:'):
                concepts_text = line.split('CONCEPTS:')[1].strip()
                if concepts_text and concepts_text != '[]':
                    concepts = [c.strip().strip('"\'') for c in concepts_text.strip('[]').split(',') if c.strip()]
            elif line.startswith('DEMOGRAPHICS:'):
                demo_text = line.split('DEMOGRAPHICS:')[1].strip()
                if demo_text and demo_text != '[]':
                    demographics = [d.strip().strip('"\'') for d in demo_text.strip('[]').split(',') if d.strip()]
        
        # Enhanced scenario determination
        scenario = _determine_extraction_scenario(entities, relationships, concepts, demographics)
        
        logger.info("enhanced_entity_extraction_completed",
                   entities_count=len(entities),
                   relationships_count=len(relationships),
                   concepts_count=len(concepts),
                   demographics_count=len(demographics),
                   scenario=scenario)
        
        return EntityExtraction(
            entities=entities,
            relationships=relationships,
            concepts=concepts,
            demographics=demographics,
            scenario=scenario
        )
        
    except Exception as e:
        logger.error("enhanced_entity_extraction_failed", error=str(e))
        return EntityExtraction(
            entities=[],
            relationships=[],
            concepts=[],
            demographics=[],
            scenario="EXTRACTION_ERROR"
        )

@tool
def generate_dynamic_cypher_query(extraction: EntityExtraction, original_query: str, llm) -> List[Dict[str, Any]]:
    """
    Generate dynamic Cypher queries using LLM based on extracted entities and demographics.
    Creates contextually appropriate Neo4j queries for different medical scenarios.
    """
    logger.info("dynamic_cypher_generation_started", scenario=extraction.scenario)
    
    try:
        # Create dynamic Cypher generation prompt
        cypher_prompt = f"""
You are a Neo4j Cypher query expert for medical knowledge graphs. Generate appropriate Cypher queries based on the extracted entities and the original user query.

EXTRACTED ENTITIES: {extraction.entities}
RELATIONSHIPS: {extraction.relationships}
CONCEPTS: {extraction.concepts}
DEMOGRAPHICS: {extraction.demographics}
SCENARIO: {extraction.scenario}

<USER_QUERY>{original_query}</USER_QUERY>

DATABASE SCHEMA:
- Nodes: Patient (gender, age, patient_id), Finding (finding_label, name), Image, FollowUp
- Relationships: Patient-[:HAS_FINDING]->Finding

CRITICAL RULES FOR QUERY GENERATION:
1. Use finding_label property (NOT label) for Finding nodes
2. Use CONTAINS for partial text matching: f.finding_label CONTAINS 'term'
3. Use COUNT(DISTINCT p) to avoid duplicate patient counts
4. For demographics: p.gender = 'M'/'F', p.age conditions
5. Always join Patient and Finding via HAS_FINDING relationship when needed
6. Generate queries specific to the extracted entities and demographics above
7. **COMPOUND FINDINGS**: When query mentions multiple conditions with "and" (e.g., "Consolidation and Effusion"),
   generate queries that find patients having BOTH conditions separately:
   MATCH (p:Patient)-[:HAS_FINDING]->(f1:Finding), (p)-[:HAS_FINDING]->(f2:Finding)
   WHERE toLower(f1.finding_label) CONTAINS 'condition1' AND toLower(f2.finding_label) CONTAINS 'condition2'
8. **AVOID COMPOUND LABELS**: Do NOT look for single labels like 'Consolidation and Effusion' - treat as separate conditions

Based on the EXTRACTED ENTITIES and DEMOGRAPHICS above, generate 1-3 optimized Cypher queries that directly answer the user's question.

Return each query on a new line starting with "QUERY:".
"""

        response = secure_llm_interaction(
            llm=llm,
            template=cypher_prompt,
            user_input=""
        )
        
        # Parse the response to extract Cypher queries
        queries = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('QUERY:'):
                cypher = line.replace('QUERY:', '').strip()
                queries.append({
                    "cypher": cypher,
                    "description": f"Dynamic query for {extraction.scenario}",
                    "parameters": {}
                })
            elif line.upper().startswith('MATCH') or line.upper().startswith('RETURN'):
                queries.append({
                    "cypher": line,
                    "description": f"Dynamic query for {extraction.scenario}",
                    "parameters": {}
                })
        
        # Return empty list if no queries found
        if not queries:
            logger.warning("no_dynamic_cypher_queries_generated")
            return []
        
        logger.info("dynamic_cypher_generation_completed", queries_count=len(queries))
        return queries
        
    except Exception as e:
        logger.error("dynamic_cypher_generation_failed", error=str(e))
        return []

@tool
def calculate_dynamic_relevance_score(triples: List[Dict[str, Any]], original_query: str, extraction: EntityExtraction, llm) -> float:
    """
    Calculate dynamic relevance score using LLM for contextual medical assessment.
    Provides intelligent scoring based on medical relevance and query intent.
    """
    logger.info("dynamic_relevance_scoring_started", triples_count=len(triples))
    
    if not triples:
        return 0.0
    
    try:
        # Create sample of triples for LLM evaluation (limit to avoid context overflow)
        sample_triples = triples[:5]
        triple_summaries = []
        
        for triple in sample_triples:
            summary = f"Subject: {triple.get('subject', 'N/A')}, Predicate: {triple.get('predicate', 'N/A')}, Object: {triple.get('object', 'N/A')}"
            triple_summaries.append(summary)
        
        relevance_prompt = f"""
You are a medical relevance expert. Evaluate how well the retrieved graph data answers the original medical query.

ORIGINAL QUERY: {original_query}
EXTRACTED ENTITIES: {extraction.entities}
DEMOGRAPHICS: {extraction.demographics}

RETRIEVED GRAPH DATA:
{chr(10).join(triple_summaries)}

<USER_QUERY>Rate the relevance of this data to the query on a scale of 0.0 to 1.0</USER_QUERY>

Consider:
1. How well the data matches the extracted entities
2. Whether demographic constraints are satisfied
3. Medical accuracy and completeness
4. Direct relevance to the question asked

Respond with only a number between 0.0 and 1.0:
"""
        
        response = secure_llm_interaction(
            llm=llm,
            template=relevance_prompt,
            user_input=""
        )
        
        # Parse score from response
        import re
        score_match = re.search(r'(\d+\.?\d*)', response.strip())
        if score_match:
            score = float(score_match.group(1))
            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
        else:
            # Return basic score if parsing fails
            score = 0.5
        
        logger.info("dynamic_relevance_scoring_completed", score=score)
        return score
        
    except Exception as e:
        logger.warning("dynamic_relevance_scoring_failed", error=str(e))
        return 0.5
@tool
def execute_dynamic_neo4j_queries(extraction: EntityExtraction, neo4j_driver, original_query: str, llm) -> GraphQueryResult:
    """
    Execute dynamic Neo4j Cypher queries based on extracted entities, relationships, and demographics.
    Builds adaptive queries for different medical scenarios and entity combinations.
    """
    logger.info("execute_dynamic_neo4j_queries_started",
               scenario=extraction.scenario,
               entities_count=len(extraction.entities),
               demographics_count=len(extraction.demographics))
    
    if not neo4j_driver:
        logger.warning("no_neo4j_driver_available")
        return GraphQueryResult(
            triples=[],
            queries_executed=0,
            scenario_used=extraction.scenario,
            total_found=0
        )
    
    triples = []
    queries_executed = 0
    
    try:
        # Generate dynamic queries using LLM
        cypher_queries = generate_dynamic_cypher_query(extraction, original_query, llm)
        
        with neo4j_driver.session() as session:
            for query_info in cypher_queries:
                try:
                    cypher = query_info.get('cypher', '')
                    parameters = query_info.get('parameters', {})
                    
                    logger.debug("executing_dynamic_cypher", cypher=cypher)
                    
                    result = session.run(cypher, parameters)
                    queries_executed += 1
                    
                    for record in result:
                        triple = _create_dynamic_triple_from_record(record, query_info, original_query)
                        triples.append(triple)
                    
                except Exception as e:
                    logger.warning("dynamic_cypher_execution_failed", cypher=cypher, error=str(e))
                    continue
        
        logger.info("execute_dynamic_neo4j_queries_completed",
                   scenario_used=extraction.scenario,
                   queries_executed=queries_executed,
                   triples_found=len(triples))
        
        return GraphQueryResult(
            triples=triples,
            queries_executed=queries_executed,
            scenario_used=extraction.scenario,
            total_found=len(triples)
        )
        
    except Exception as e:
        logger.error("dynamic_neo4j_queries_failed", error=str(e), scenario=extraction.scenario)
        return GraphQueryResult(
            triples=[],
            queries_executed=0,
            scenario_used=extraction.scenario,
            total_found=0
        )

# ==================== END ENHANCED GRAPH SEARCH TOOLS ====================

# Factory function for easy instantiation
def create_simple_agentic_workflow(llm: AzureChatOpenAI, vector_store, graph_store) -> SimpleAgenticWorkflow:
    """Create a simple agentic workflow with all required components"""
    logger.info("creating_simple_agentic_workflow")
    workflow = SimpleAgenticWorkflow(llm, vector_store, graph_store)
    logger.info("simple_agentic_workflow_created")
    return workflow

# ==================== TOOL IMPLEMENTATIONS FOR GRAPH RAG ====================
# These are the @tool decorated versions for external tool registration

@tool
def extract_entities_and_relationships_tool(query: str, llm) -> EntityExtraction:
    """
    Tool for extracting entities and relationships using comprehensive prompt template.
    Converts the AgenticGraphRAGAgent method to a tool for external use.
    """
    # Import here to avoid circular imports
    from core.input_sanitization import (
        detect_prompt_injection,
        secure_llm_interaction
    )
    
    logger.info("tool_entity_extraction_started", query_length=len(query))
    
    # Step 1: Detect prompt injection attempts
    if detect_prompt_injection(query):
        logger.warning("prompt_injection_blocked_in_tool", query_snippet=query[:50])
        return EntityExtraction(
            entities=[],
            relationships=[],
            concepts=[],
            demographics=[],
            scenario="PROMPT_INJECTION_DETECTED"
        )
    
    # Step 2: Use comprehensive entity extraction template
    comprehensive_template = """
You are a medical knowledge graph entity extraction expert. Analyze the following medical query and extract ALL relevant information in the specified format.

<USER_QUERY>{user_query}</USER_QUERY>

Extract the following information:

1. MEDICAL ENTITIES: Diseases, symptoms, treatments, medications, procedures, body parts, medical conditions
2. RELATIONSHIPS: Connection words (has, causes, treats, affects, diagnosed_with, prescribed_for, related_to)
3. MEDICAL CONCEPTS: Medical domains/specialties (cardiology, oncology, neurology, pediatrics, etc.)
4. DEMOGRAPHICS: Age, gender, ethnicity, location, patient characteristics

Format your response EXACTLY as follows:
ENTITIES: [entity1, entity2, entity3]
RELATIONSHIPS: [relationship1, relationship2]
CONCEPTS: [concept1, concept2]
DEMOGRAPHICS: [demographic1, demographic2]

Be comprehensive and include ALL relevant medical terms, even if they seem obvious.
If a category has no items, use empty brackets: []

Examples:
- "chest pain in elderly women" -> ENTITIES: [chest pain], RELATIONSHIPS: [has], CONCEPTS: [cardiology], DEMOGRAPHICS: [elderly, women]
- "diabetes treatment options" -> ENTITIES: [diabetes, treatment], RELATIONSHIPS: [treats], CONCEPTS: [endocrinology], DEMOGRAPHICS: []
"""
    
    try:
        # Use secure LLM interaction
        response = secure_llm_interaction(
            llm=llm,
            template=comprehensive_template,
            user_input=query
        )
        
        # Parse the response using helper function
        def parse_list_from_string(list_str: str) -> List[str]:
            """Parse a list from string format like '[item1, item2, item3]'"""
            if not list_str or list_str.strip() == '[]':
                return []
            
            # Remove brackets and split by comma
            cleaned = list_str.strip()
            if cleaned.startswith('[') and cleaned.endswith(']'):
                cleaned = cleaned[1:-1]
            
            items = [item.strip().strip("'\"") for item in cleaned.split(',') if item.strip()]
            return [item for item in items if item]  # Remove empty items
        
        entities = []
        relationships = []
        concepts = []
        demographics = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('ENTITIES:'):
                entities_str = line.split(':', 1)[1].strip()
                entities = parse_list_from_string(entities_str)
            elif line.startswith('RELATIONSHIPS:'):
                relationships_str = line.split(':', 1)[1].strip()
                relationships = parse_list_from_string(relationships_str)
            elif line.startswith('CONCEPTS:'):
                concepts_str = line.split(':', 1)[1].strip()
                concepts = parse_list_from_string(concepts_str)
            elif line.startswith('DEMOGRAPHICS:'):
                demographics_str = line.split(':', 1)[1].strip()
                demographics = parse_list_from_string(demographics_str)
        
        # Determine scenario based on extracted content
        scenario = "GENERAL_QUERY"
        if demographics:
            scenario = "DEMOGRAPHIC_QUERY"
        elif any(concept in ['cardiology', 'oncology', 'neurology'] for concept in concepts):
            scenario = "SPECIALIZED_QUERY"
        
        result = EntityExtraction(
            entities=entities,
            relationships=relationships,
            concepts=concepts,
            demographics=demographics,
            scenario=scenario
        )
        
        logger.info("tool_entity_extraction_completed",
                   entities_count=len(entities),
                   relationships_count=len(relationships),
                   concepts_count=len(concepts),
                   demographics_count=len(demographics),
                   scenario=scenario)
        
        return result
        
    except Exception as e:
        logger.error("tool_entity_extraction_failed", error=str(e))
        return EntityExtraction(
            entities=[],
            relationships=[],
            concepts=[],
            demographics=[],
            scenario="EXTRACTION_ERROR"
        )

@tool
def generate_cypher_query_tool(extraction: EntityExtraction, original_query: str, llm) -> List[Dict[str, Any]]:
    """
    Tool for generating Cypher queries using comprehensive prompt template.
    Converts the AgenticGraphRAGAgent method to a tool for external use.
    """
    logger.info("tool_cypher_generation_started", 
               scenario=extraction.scenario, 
               entities_count=len(extraction.entities))
    
    # Import here to avoid circular imports
    from core.input_sanitization import secure_llm_interaction
    
    try:
        # Comprehensive Cypher generation template
        cypher_template = """
You are a Neo4j Cypher query expert for medical databases. Generate optimized Cypher queries based on the extracted entities and demographics.

ORIGINAL QUERY: {original_query}
EXTRACTED ENTITIES: {entities}
DEMOGRAPHICS: {demographics}
MEDICAL CONCEPTS: {concepts}
RELATIONSHIPS: {relationships}

DATABASE SCHEMA:
- Patient nodes: properties include gender ('M'/'F'), age (integer), patient_id
- Finding nodes: properties include finding_label, name
- Relationships: (Patient)-[:HAS_FINDING]->(Finding)

Generate 1-3 Cypher queries that would answer the original question. Focus on:
1. Demographic filtering (age, gender)
2. Finding/condition matching
3. Count aggregations when appropriate

Format each query as:
QUERY_TYPE: [descriptive name]
CYPHER: [cypher query]
DESCRIPTION: [what this query does]

RULES FOR DYNAMIC GENERATION:
1. Generate queries based on the specific EXTRACTED data above
2. Use finding_label property for Finding nodes
3. Use CONTAINS for partial matching
4. Use COUNT(p) or COUNT(DISTINCT p) for counting
5. Include demographic filters when present in extraction
6. Focus on the specific entities and concepts identified
"""
        
        # Format the template with extracted data
        formatted_template = cypher_template.format(
            original_query=original_query,
            entities=extraction.entities,
            demographics=extraction.demographics,
            concepts=extraction.concepts,
            relationships=extraction.relationships
        )
        
        # Use secure LLM interaction
        response = secure_llm_interaction(
            llm=llm,
            template=formatted_template,
            user_input=""
        )
        
        # Parse the response to extract queries
        queries = []
        current_query = {}
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('QUERY_TYPE:'):
                if current_query:
                    queries.append(current_query)
                current_query = {'query_type': line.split(':', 1)[1].strip()}
            elif line.startswith('CYPHER:'):
                current_query['cypher'] = line.split(':', 1)[1].strip()
            elif line.startswith('DESCRIPTION:'):
                current_query['description'] = line.split(':', 1)[1].strip()
        
        # Add the last query
        if current_query:
            queries.append(current_query)
        
        # Return empty list if no queries parsed
        if not queries:
            logger.warning("no_cypher_queries_parsed_from_llm_response")
            return []
        
        logger.info("tool_cypher_generation_completed", queries_count=len(queries))
        return queries
        
    except Exception as e:
        logger.error("tool_cypher_generation_failed", error=str(e))
        return []
