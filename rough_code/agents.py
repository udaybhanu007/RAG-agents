from typing import List, Dict, Any, Optional
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from qdrant_client import QdrantClient
from workflow_state import WorkflowState
from observability import observability
from logging_config import get_logger

logger = get_logger("agents")


# Pydantic models for structured outputs
class QueryAnalysis(BaseModel):
    """Simple query analysis focused on routing needs"""
    intent: str = Field(description="Intent: FACTUAL, RELATIONAL, or ANALYTICAL")
    entity_count: int = Field(description="Estimated number of entities (1-3+)")
    has_relationships: bool = Field(description="Whether query asks about relationships")


class RoutingDecision(BaseModel):
    """Routing decision with justification"""
    route: str = Field(description="Chosen route: vector, graph, both, or none")
    confidence: str = Field(description="Confidence level: HIGH, MEDIUM, or LOW")
    reasoning: str = Field(description="Brief explanation for the routing decision")


class VectorSearchResult(BaseModel):
    """Vector search result with document information"""
    documents: List[Dict[str, Any]] = Field(description="Retrieved documents with scores")
    total_found: int = Field(description="Total number of documents found")
    precision_score: Optional[float] = Field(description="Retrieval precision if relevance labels available")


class RerankedResult(BaseModel):
    """Reranked documents result"""
    documents: List[Dict[str, Any]] = Field(description="Reranked documents in relevance order")
    reranking_applied: bool = Field(description="Whether reranking was successfully applied")


class EntityExtraction(BaseModel):
    """Entity extraction result"""
    entities: List[str] = Field(description="Extracted medical entities")
    relationships: List[str] = Field(description="Extracted relationship indicators")
    concepts: List[str] = Field(description="Extracted medical concepts/domains")
    scenario: str = Field(description="Identified query scenario type")


class GraphQueryResult(BaseModel):
    """Graph query execution result"""
    triples: List[Dict[str, Any]] = Field(description="Retrieved knowledge graph triples")
    queries_executed: int = Field(description="Number of Cypher queries executed")
    scenario_used: str = Field(description="Query scenario that was applied")


class QueryValidation(BaseModel):
    """Simple query validation for medical relevance"""
    is_medical: bool = Field(description="Whether query is medical/healthcare related")
    quick_response: Optional[str] = Field(description="Response for non-medical queries")


# Function calling tools for OrchestratorAgent
@tool
def validate_medical_relevance(query: str, llm) -> QueryValidation:
    """
    Simple validation to check if query is medical/healthcare related using LLM.
    
    Args:
        query: The user query to validate
        llm: LLM instance for classification
        
    Returns:
        QueryValidation with medical relevance assessment
    """
    
    validation_prompt = """
You are a medical query classifier. Determine if this query is medical/healthcare related.

MEDICAL/HEALTHCARE queries include:
- Medical conditions, diseases, symptoms, treatments
- Anatomy, physiology, medications, procedures
- Healthcare systems, medical diagnostics
- Patient care, clinical scenarios

NON-MEDICAL queries include:
- General greetings, technology, programming
- Sports, entertainment, travel, cooking
- Business, finance, academic (non-medical)

Query: "{query}"

Respond with only:
MEDICAL or NON_MEDICAL
"""

    try:
        response = llm.invoke(validation_prompt.format(query=query))
        content = response.content.strip().upper()
        
        is_medical = "MEDICAL" in content
        
        quick_response = None
        if not is_medical:
            quick_response = "I'm a medical knowledge assistant specialized in healthcare topics. Please ask me questions about medical conditions, treatments, symptoms, or other healthcare-related matters."
        
        return QueryValidation(
            is_medical=is_medical,
            quick_response=quick_response
        )
        
    except Exception as e:
        logger.warning("llm_validation_failed", error=str(e))
        # Conservative fallback - assume medical to avoid blocking valid queries
        return QueryValidation(
            is_medical=True,
            quick_response=None
        )


@tool
def analyze_query_characteristics(query: str) -> QueryAnalysis:
    """
    Simple query analysis focused on routing decisions.
    Uses minimal heuristics for fast, reliable routing.
    
    Args:
        query: The user query to analyze
        
    Returns:
        QueryAnalysis with intent, entity count, and relationship detection
    """
    query_lower = query.lower()
    
    # Simple intent detection using key patterns
    if any(word in query_lower for word in ["compare", "versus", "vs", "difference", "similar", "contrast"]):
        intent = "ANALYTICAL"
    elif any(word in query_lower for word in ["relationship", "connect", "between", "relate", "link", "associate", "affect", "cause"]):
        intent = "RELATIONAL"
    else:
        intent = "FACTUAL"  # Default for most queries
    
    # Simple entity count estimation
    # Count potential medical/technical terms (capitalized words, medical suffixes)
    import re
    
    # Look for medical-like terms and proper nouns
    medical_suffixes = re.findall(r'\b\w+(megaly|itis|osis|emia|pathy)\b', query_lower)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
    common_medical = ["chest", "lung", "heart", "brain", "x-ray", "ct", "mri", "scan"]
    medical_words = [word for word in common_medical if word in query_lower]
    
    # Combine all potential entities
    all_entities = medical_suffixes + proper_nouns + medical_words
    entity_count = min(3, max(1, len(set(all_entities))))  # Cap at 3, minimum 1
    
    # Simple relationship detection
    relationship_indicators = ["relationship", "connect", "between", "relate", "link", "associate", "depend", "affect", "cause"]
    has_relationships = any(indicator in query_lower for indicator in relationship_indicators)
    
    return QueryAnalysis(
        intent=intent,
        entity_count=entity_count,
        has_relationships=has_relationships
    )


# Function calling tools for GraphRAGAgent
@tool
def extract_entities_from_query(query: str, llm) -> EntityExtraction:
    """
    Extract entities, relationships, and concepts from medical query.
    
    Args:
        query: The medical query to analyze
        llm: LLM instance for entity extraction
        
    Returns:
        EntityExtraction with structured results
    """
    extraction_prompt = """
    Extract medical entities, relationships, and concepts from this query.

    ENTITY TYPES:
    - MEDICAL CONDITIONS: diseases, disorders, pathologies
    - ANATOMICAL STRUCTURES: body parts, organs  
    - MEDICAL PROCEDURES: tests, imaging, treatments
    - CLINICAL FINDINGS: symptoms, signs
    - CONTEXTUAL: severity, location, timing

    RELATIONSHIP INDICATORS:
    - CAUSATIVE: "causes", "leads to", "results in"
    - ASSOCIATED: "associated with", "related to", "linked to"  
    - DIAGNOSTIC: "indicates", "suggests", "shows"
    - LOCATIONAL: "located in", "affects", "involves"

    OUTPUT FORMAT (REQUIRED):
    ENTITIES: [entity1, entity2, entity3]
    RELATIONSHIPS: [relationship1, relationship2]
    CONCEPTS: [concept1, concept2]
    
    Query: {query}
    """
    
    # Get LLM extraction
    response = llm.invoke(extraction_prompt.format(query=query))
    
    # Parse response
    entities, relationships, concepts = [], [], []
    
    try:
        lines = response.content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('ENTITIES:'):
                entities_text = line.split('ENTITIES:')[1].strip()
                if entities_text and entities_text != '[]':
                    entities = [e.strip().strip('"\'') for e in entities_text.strip('[]').split(',')]
            elif line.startswith('RELATIONSHIPS:'):
                rel_text = line.split('RELATIONSHIPS:')[1].strip()
                if rel_text and rel_text != '[]':
                    relationships = [r.strip().strip('"\'') for r in rel_text.strip('[]').split(',')]
            elif line.startswith('CONCEPTS:'):
                concepts_text = line.split('CONCEPTS:')[1].strip()
                if concepts_text and concepts_text != '[]':
                    concepts = [c.strip().strip('"\'') for c in concepts_text.strip('[]').split(',')]
    except Exception as e:
        logger.warning("entity_parsing_failed", error=str(e))
    
    # Determine scenario
    if len(entities) >= 2 and relationships:
        scenario = "MULTI_ENTITY_WITH_RELATIONSHIPS"
    elif len(entities) >= 2:
        scenario = "MULTI_ENTITY_NO_RELATIONSHIPS"
    elif len(entities) == 1 and relationships:
        scenario = "SINGLE_ENTITY_WITH_RELATIONSHIPS"
    elif len(entities) == 1:
        scenario = "SINGLE_ENTITY_NO_RELATIONSHIPS"
    else:
        scenario = "CONCEPTS_ONLY"
    
    return EntityExtraction(
        entities=entities,
        relationships=relationships,
        concepts=concepts,
        scenario=scenario
    )


@tool
def execute_graph_queries(extraction: EntityExtraction, neo4j_driver) -> GraphQueryResult:
    """
    Execute Cypher queries based on entity extraction results.
    
    Args:
        extraction: EntityExtraction with entities, relationships, concepts
        neo4j_driver: Neo4j driver instance
        
    Returns:
        GraphQueryResult with retrieved triples
    """
    entities = extraction.entities
    relationships = extraction.relationships
    concepts = extraction.concepts
    scenario = extraction.scenario
    
    # Build queries based on scenario
    queries = []
    
    if scenario == "MULTI_ENTITY_WITH_RELATIONSHIPS":
        # Entity pair relationships
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                query = f"""
                MATCH (a)-[r]-(b) 
                WHERE (toLower(a.name) CONTAINS toLower('{entities[i]}') OR toLower(a.title) CONTAINS toLower('{entities[i]}'))
                AND (toLower(b.name) CONTAINS toLower('{entities[j]}') OR toLower(b.title) CONTAINS toLower('{entities[j]}'))
                RETURN a.name as subject, type(r) as predicate, b.name as object
                LIMIT 5
                """
                queries.append(query)
        
        # Relationship-specific queries
        if relationships:
            primary_entity = entities[0]
            for relationship in relationships[:2]:  # Limit to avoid query explosion
                query = f"""
                MATCH (a)-[r]-(b)
                WHERE (toLower(a.name) CONTAINS toLower('{primary_entity}') OR toLower(a.title) CONTAINS toLower('{primary_entity}'))
                AND (toLower(type(r)) CONTAINS toLower('{relationship}') OR toLower(r.type) CONTAINS toLower('{relationship}'))
                RETURN a.name as subject, type(r) as predicate, b.name as object
                LIMIT 5
                """
                queries.append(query)
    
    elif scenario == "MULTI_ENTITY_NO_RELATIONSHIPS":
        # Simple entity pair connections
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                query = f"""
                MATCH (a)-[r]-(b) 
                WHERE (toLower(a.name) CONTAINS toLower('{entities[i]}') OR toLower(a.title) CONTAINS toLower('{entities[i]}'))
                AND (toLower(b.name) CONTAINS toLower('{entities[j]}') OR toLower(b.title) CONTAINS toLower('{entities[j]}'))
                RETURN a.name as subject, type(r) as predicate, b.name as object
                LIMIT 8
                """
                queries.append(query)
    
    elif scenario == "SINGLE_ENTITY_WITH_RELATIONSHIPS":
        entity = entities[0]
        for relationship in relationships:
            query = f"""
            MATCH (a)-[r]-(b)
            WHERE (toLower(a.name) CONTAINS toLower('{entity}') OR toLower(a.title) CONTAINS toLower('{entity}'))
            AND (toLower(type(r)) CONTAINS toLower('{relationship}') OR toLower(r.type) CONTAINS toLower('{relationship}'))
            RETURN a.name as subject, type(r) as predicate, b.name as object
            LIMIT 8
            """
            queries.append(query)
    
    elif scenario == "SINGLE_ENTITY_NO_RELATIONSHIPS":
        entity = entities[0]
        # Entity properties and connections
        queries = [
            f"""
            MATCH (n) 
            WHERE toLower(n.name) CONTAINS toLower('{entity}') OR toLower(n.title) CONTAINS toLower('{entity}')
            RETURN n.name as subject, 'has_property' as predicate, n as object
            LIMIT 5
            """,
            f"""
            MATCH (a)-[r]-(b)
            WHERE toLower(a.name) CONTAINS toLower('{entity}') OR toLower(a.title) CONTAINS toLower('{entity}')
            RETURN a.name as subject, type(r) as predicate, b.name as object
            LIMIT 8
            """
        ]
    
    else:  # CONCEPTS_ONLY
        for concept in concepts:
            query = f"""
            MATCH (n) 
            WHERE toLower(n.category) CONTAINS toLower('{concept}') 
            OR toLower(n.domain) CONTAINS toLower('{concept}')
            OR toLower(n.specialty) CONTAINS toLower('{concept}')
            RETURN n.name as subject, 'belongs_to_concept' as predicate, '{concept}' as object
            LIMIT 10
            """
            queries.append(query)
    
    # Execute queries and collect results
    triples = []
    
    with neo4j_driver.session() as session:
        for cypher_query in queries:
            try:
                result = session.run(cypher_query)
                for record in result:
                    triple = {
                        "subject": record.get("subject", ""),
                        "predicate": record.get("predicate", ""),
                        "object": record.get("object", ""),
                        "metadata": dict(record.items()),
                        "source": "knowledge_graph",
                        "query": cypher_query
                    }
                    triples.append(triple)
            except Exception as e:
                logger.warning("cypher_query_failed", query=cypher_query, error=str(e))
    
    # Log if no triples were found
    if not triples:
        logger.info(
            "graph_no_results_found",
            scenario=scenario,
            entities_count=len(entities),
            relationships_count=len(relationships),
            concepts_count=len(concepts),
            queries_executed=len(queries)
        )
    
    return GraphQueryResult(
        triples=triples,
        queries_executed=len(queries),
        scenario_used=scenario
    )


# Function calling tools for VectorRAGAgent
@tool
def assess_document_relevance(query: str, document_content: str, llm) -> bool:
    """
    Assess if a document is relevant to the query using LLM.
    
    Args:
        query: The search query
        document_content: Content of the retrieved document
        llm: LLM instance for relevance assessment
        
    Returns:
        Boolean indicating if document is relevant
    """
    relevance_prompt = f"""
    Query: {query}
    
    Document: {document_content[:500]}...
    
    Is this document relevant to answering the query? Consider if the document contains information that directly addresses or helps answer the question.
    
    Answer only: YES or NO
    """
    
    try:
        response = llm.invoke(relevance_prompt)
        return response.content.strip().upper() == "YES"
    except Exception as e:
        logger.warning("relevance_assessment_failed", error=str(e))
        return False  # Conservative approach - assume not relevant if assessment fails


@tool
def perform_vector_search(query: str, qdrant_client, embeddings, llm=None, collection_name: str = "documents", limit: int = 10, score_threshold: float = 0.6) -> VectorSearchResult:
    """
    Perform semantic search using Qdrant vector database with dynamic relevance assessment.
    
    Args:
        query: The search query
        qdrant_client: Qdrant client instance
        embeddings: Azure OpenAI embeddings instance
        llm: LLM instance for relevance assessment (optional)
        collection_name: Name of the Qdrant collection
        limit: Maximum number of documents to retrieve
        score_threshold: Minimum similarity score threshold
        
    Returns:
        VectorSearchResult with documents and precision score
    """
    # Generate query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Search in Qdrant
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False
    )
    
    # Convert results to standard format
    documents = []
    for result in search_results:
        doc = {
            "id": result.id,
            "content": result.payload.get("content", ""),
            "metadata": result.payload.get("metadata", {}),
            "score": float(result.score),
            "source": "vector_store"
        }
        documents.append(doc)
    
    # Calculate statistics
    total_found = len(documents)
    
    # Log if no documents were found
    if not documents:
        logger.info(
            "vector_no_results_found",
            collection_name=collection_name,
            score_threshold=score_threshold,
            limit=limit
        )
    
    # Calculate precision using dynamic relevance assessment if LLM is available
    precision_score = None
    if llm and documents:
        relevant_count = 0
        for doc in documents:
            is_relevant = assess_document_relevance.invoke({
                "query": query,
                "document_content": doc["content"],
                "llm": llm
            })
            if is_relevant:
                relevant_count += 1
        
        precision_score = relevant_count / total_found if total_found > 0 else 0.0
        
        # Log precision metrics
        logger.info(
            "vector_search_precision_calculated",
            query_length=len(query),
            total_documents=total_found,
            relevant_documents=relevant_count,
            precision_score=precision_score,
            collection_name=collection_name,
            score_threshold=score_threshold
        )
    
    return VectorSearchResult(
        documents=documents,
        total_found=total_found,
        precision_score=precision_score
    )


@tool
def rerank_documents_by_relevance(query: str, documents: List[Dict[str, Any]], llm) -> RerankedResult:
    """
    Rerank documents using LLM for better relevance ordering.
    
    Args:
        query: The original search query
        documents: List of documents to rerank
        llm: LLM instance for reranking
        
    Returns:
        RerankedResult with reordered documents
    """
    if not documents or len(documents) <= 1:
        return RerankedResult(documents=documents, reranking_applied=False)
    
    try:
        # Create reranking prompt
        doc_list = "\n".join([f"{i+1}. {doc['content'][:200]}..." for i, doc in enumerate(documents)])
        
        rerank_prompt = f"""
        Query: {query}
        
        Rank these documents by relevance (1 = most relevant):
        
        {doc_list}
        
        Return only numbers separated by commas (e.g., 3,1,4,2):
        """
        
        # Get LLM ranking
        response = llm.invoke(rerank_prompt)
        rankings = [int(x.strip()) - 1 for x in response.content.strip().split(',')]
        
        # Validate and reorder
        if len(rankings) == len(documents) and all(0 <= r < len(documents) for r in rankings):
            reranked_docs = [documents[i] for i in rankings]
            return RerankedResult(documents=reranked_docs, reranking_applied=True)
        
    except Exception as e:
        logger.warning("reranking_failed", error=str(e))
    
    # Return original order if reranking fails
    return RerankedResult(documents=documents, reranking_applied=False)


@tool
def determine_optimal_route(analysis: QueryAnalysis) -> RoutingDecision:
    """
    Mutually exclusive routing logic based on intent and entities.
    Uses if-elif-else structure to ensure only one rule applies.
    
    Args:
        analysis: QueryAnalysis with intent, entity_count, has_relationships
        
    Returns:
        RoutingDecision with route and reasoning
    """
    # Mutually exclusive routing rules (ordered by priority)
    
    # Rule 1: Analytical queries always need comprehensive search
    if analysis.intent == "ANALYTICAL":
        return RoutingDecision(
            route="both",
            confidence="HIGH",
            reasoning="Analytical query requires comprehensive search"
        )
    
    # Rule 2: Relational queries with multiple entities prefer graph
    elif analysis.intent == "RELATIONAL" and analysis.entity_count >= 2:
        return RoutingDecision(
            route="graph",
            confidence="HIGH",
            reasoning="Relational query with multiple entities - graph optimal"
        )
    
    # Rule 3: Relational queries with single entity still prefer graph
    elif analysis.intent == "RELATIONAL":
        return RoutingDecision(
            route="graph",
            confidence="MEDIUM",
            reasoning="Relational query - graph preferred for relationships"
        )
    
    # Rule 4: Multiple entities (3+) need comprehensive search
    elif analysis.entity_count >= 3:
        return RoutingDecision(
            route="both",
            confidence="MEDIUM",
            reasoning="Multiple entities require comprehensive search"
        )
    
    # Rule 5: Relationship indicators (non-relational intent) prefer graph
    elif analysis.has_relationships:
        return RoutingDecision(
            route="graph",
            confidence="MEDIUM",
            reasoning="Relationship indicators detected - graph preferred"
        )
    
    # Rule 6: Simple factual queries with 1-2 entities use vector
    else:
        return RoutingDecision(
            route="vector",
            confidence="HIGH",
            reasoning="Simple factual query - semantic search optimal"
        )


class OrchestratorAgent:
    """
    Enhanced Function-Calling Orchestrator Agent with Medical Query Validation
    First validates if query is medical, then routes accordingly
    Reads: state.query
    Writes: state.route, state.latency_ms["orch"], state.final_answer (for non-medical queries)
    """
    
    def __init__(self, llm: AzureChatOpenAI = None):
        self.llm = llm  # LLM for medical validation
    
    def route_query(self, state: WorkflowState) -> WorkflowState:
        """Route the query with medical validation using function calling approach"""
        
        with observability.measure_agent_performance("orch", state):
            try:
                query = state["query"]
                
                # Step 1: Validate if query is medical/healthcare related (only if LLM available)
                if self.llm:
                    validation_result = validate_medical_relevance.invoke({
                        "query": query,
                        "llm": self.llm
                    })
                    
                    # Handle non-medical queries immediately
                    if not validation_result.is_medical:
                        state["route"] = "none"
                        state["routing_analysis"] = "Non-medical query detected"
                        state["final_answer"] = validation_result.quick_response
                        state["bypass_retrieval"] = True
                        
                        logger.info(
                            "orchestrator_non_medical_query",
                            query_length=len(query),
                            trace_id=state.get('trace_id')
                        )
                        
                        return state
                
                # Step 2: For medical queries, proceed with normal routing
                # Analyze query characteristics
                analysis_result = analyze_query_characteristics.invoke({"query": query})
                
                # Determine optimal route based on analysis
                routing_result = determine_optimal_route.invoke({"analysis": analysis_result})
                
                # Extract routing information
                route = routing_result.route
                reasoning = routing_result.reasoning
                
                # Create analysis summary
                analysis = f"Intent: {analysis_result.intent}, Entities: {analysis_result.entity_count}, Relationships: {analysis_result.has_relationships}"
                
                # Update state with routing information
                state["route"] = route
                state["routing_analysis"] = analysis
                state["bypass_retrieval"] = False
                
                logger.info(
                    "orchestrator_medical_routing",
                    route=route,
                    reasoning=reasoning,
                    analysis=analysis,
                    query_length=len(query),
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("orchestrator_function_calling_error", error=str(e), trace_id=state.get('trace_id'))
                state["errors"] = state.get("errors", []) + [f"Orchestrator function calling error: {str(e)}"]
                state["route"] = "both"  # Safe fallback
                state["routing_analysis"] = "Error during analysis"
                state["bypass_retrieval"] = False
                return state


class VectorRAGAgent:
    """
    Function-Calling Vector RAG Agent
    Performs semantic search and reranking using direct tool execution
    Reads: state.query
    Writes: state.vector_docs, state.latency_ms["vec"]
    """
    
    def __init__(self, qdrant_client: QdrantClient, embeddings: AzureOpenAIEmbeddings, 
                 collection_name: str = "documents", llm: AzureChatOpenAI = None):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.llm = llm
    
    def retrieve_documents(self, state: WorkflowState) -> WorkflowState:
        """Retrieve documents using function calling approach"""
        
        with observability.measure_agent_performance("vec", state):
            try:
                query = state["query"]
                
                # Step 1: Perform vector search using tool
                search_result = perform_vector_search.invoke({
                    "query": query,
                    "qdrant_client": self.qdrant_client,
                    "embeddings": self.embeddings,
                    "llm": self.llm,
                    "collection_name": self.collection_name,
                    "limit": 10,
                    "score_threshold": 0.6
                })
                
                documents = search_result.documents
                
                # Step 2: Optional reranking if LLM available and enough real documents
                if self.llm and len(documents) > 3:
                    rerank_result = rerank_documents_by_relevance.invoke({
                        "query": query,
                        "documents": documents,
                        "llm": self.llm
                    })
                    documents = rerank_result.documents
                    reranking_applied = rerank_result.reranking_applied
                else:
                    reranking_applied = False
                
                # Update state
                state["vector_docs"] = documents
                
                logger.info(
                    "vector_retrieval_function_calling",
                    documents_retrieved=len(documents),
                    precision_score=search_result.precision_score,
                    reranking_applied=reranking_applied,
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("vector_rag_function_calling_error", error=str(e), trace_id=state.get('trace_id'))
                state["errors"] = state.get("errors", []) + [f"Vector RAG error: {str(e)}"]
                state["vector_docs"] = []
                return state


class GraphRAGAgent:
    """
    Function-Calling Graph RAG Agent
    Performs entity extraction and graph queries using direct tool execution
    Reads: state.query
    Writes: state.graph_triples, state.latency_ms["graph"]
    """
    
    def __init__(self, neo4j_driver, llm: AzureChatOpenAI):
        self.driver = neo4j_driver
        self.llm = llm
    
    def extract_and_query(self, state: WorkflowState) -> WorkflowState:
        """Extract entities and query knowledge graph using function calling approach"""
        
        with observability.measure_agent_performance("graph", state):
            try:
                query = state["query"]
                
                # Step 1: Extract entities, relationships, and concepts
                extraction_result = extract_entities_from_query.invoke({
                    "query": query,
                    "llm": self.llm
                })
                
                # Step 2: Execute graph queries based on extraction
                graph_result = execute_graph_queries.invoke({
                    "extraction": extraction_result,
                    "neo4j_driver": self.driver
                })
                
                # Update state
                state["graph_triples"] = graph_result.triples
                
                logger.info(
                    "graph_retrieval_function_calling",
                    entities_found=len(extraction_result.entities),
                    relationships_found=len(extraction_result.relationships),
                    concepts_found=len(extraction_result.concepts),
                    scenario_used=extraction_result.scenario,
                    queries_executed=graph_result.queries_executed,
                    triples_retrieved=len(graph_result.triples),
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("graph_rag_function_calling_error", error=str(e), trace_id=state.get('trace_id'))
                state["errors"] = state.get("errors", []) + [f"Graph RAG error: {str(e)}"]
                state["graph_triples"] = []
                return state
