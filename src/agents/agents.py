from typing import List, Dict, Any, Optional, cast
import re
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever
from pydantic.v1 import BaseModel, Field
from qdrant_client import QdrantClient
from workflow_state import WorkflowState
from observability import observability
from logging_config import get_logger
from tool_governance import ToolRegistry, ToolMetadata, AgentRole, tool_registry, AccessDeniedError, SecureAgentBase
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

class HybridSearchResult(BaseModel):
    """Hybrid search result combining vector and BM25"""
    documents: List[Dict[str, Any]] = Field(description="Combined and reranked documents")
    vector_count: int = Field(description="Number of documents from vector search")
    bm25_count: int = Field(description="Number of documents from BM25 search")
    total_found: int = Field(description="Total unique documents after merging")
    search_strategy: str = Field(description="Strategy used: vector_only, bm25_only, or hybrid")
    precision_score: Optional[float] = Field(description="Hybrid precision score")


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
        content = str(response.content).strip().upper()
        
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


# Function calling tools for OrchestratorAgent
@tool
def analyze_query_characteristics(query: str, llm: AzureChatOpenAI) -> QueryAnalysis:
    """
    Dynamic query analysis using LLM for better intent classification.
    
    Args:
        query: The user query to analyze
        llm: LLM instance for dynamic analysis
        
    Returns:
        QueryAnalysis with intent, entity count, and relationship detection
    """
    
    analysis_prompt = """
Analyze this medical query and classify its characteristics:

INTENT TYPES:
- FACTUAL: Seeks specific facts, definitions, symptoms, procedures (What is...? How to...? When to...?)
- RELATIONAL: Explores connections, relationships, interactions between entities (How does X affect Y? What's the relationship between...? X vs Y in terms of...)
- ANALYTICAL: Requires comparison, evaluation, analysis of multiple aspects (Compare X and Y, Which is better, Analyze the differences, Evaluate effectiveness)

ENTITY COUNT: Count distinct medical entities, conditions, procedures, demographics, or concepts:
- Consider: patients, medical findings, conditions, procedures, timeframes, demographics
- 1: Single primary entity (but may have related sub-entities)
- 2: Two distinct main entities 
- 3: Three or more main entities

RELATIONSHIPS: Does the query ask about connections, correlations, or interactions?
IMPORTANT: Medical queries often have implicit relationships:
- Patient + findings = relationship (medical history, patient findings)
- Condition + progression = relationship (disease progression) 
- Findings + locations = relationship (anatomical relationships)
- Patient + demographics = relationship (patient characteristics)
- Time + changes = relationship (progression, evolution)

Look for these relationship indicators:
- Explicit: "relationship", "connection", "between", "affects", "causes", "leads to"
- Implicit: "history", "progression", "findings", "characteristics", "demographics", "locations", "dimensions"
- Temporal: "over time", "progression", "changes", "evolution"
- Medical context: "patient + findings", "condition + symptoms", "treatment + outcomes"

Query: "{query}"

Respond in this exact format:
INTENT: [FACTUAL|RELATIONAL|ANALYTICAL]
ENTITY_COUNT: [1|2|3]
HAS_RELATIONSHIPS: [true|false]
REASONING: [Brief explanation focusing on entities and relationships detected]
"""

    try:
        response = llm.invoke(analysis_prompt.format(query=query))
        content = str(response.content).strip()
        
        # Parse LLM response
        intent = "FACTUAL"  # Default
        entity_count = 1
        has_relationships = False
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('INTENT:'):
                intent_value = line.split('INTENT:')[1].strip()
                if intent_value in ["FACTUAL", "RELATIONAL", "ANALYTICAL"]:
                    intent = intent_value
            elif line.startswith('ENTITY_COUNT:'):
                try:
                    count_value = line.split('ENTITY_COUNT:')[1].strip()
                    entity_count = min(3, max(1, int(count_value)))
                except ValueError:
                    entity_count = 1
            elif line.startswith('HAS_RELATIONSHIPS:'):
                rel_value = line.split('HAS_RELATIONSHIPS:')[1].strip().lower()
                has_relationships = rel_value == 'true'
        
        return QueryAnalysis(
            intent=intent,
            entity_count=entity_count,
            has_relationships=has_relationships
        )
        
    except Exception as e:
        logger.warning("llm_analysis_failed", error=str(e), query=query[:100])
        # Conservative fallback with basic defaults
        return QueryAnalysis(
            intent="FACTUAL",
            entity_count=1,
            has_relationships=False
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
    
    return GraphQueryResult(
        triples=triples,
        queries_executed=len(queries),
        scenario_used=scenario
    )


# Function calling tools for VectorRAGAgent
@tool
def perform_hybrid_search(query: str, qdrant_client, embeddings, bm25_retriever=None, 
                         llm=None, collection_name: str = "documents", 
                         limit: int = 10, score_threshold: float = 0.6) -> HybridSearchResult:
    """
    Perform hybrid search combining vector similarity and BM25 keyword search.
    
    Args:
        query: The search query
        qdrant_client: Qdrant client instance
        embeddings: Embeddings instance (HuggingFace or Azure)
        bm25_retriever: BM25 retriever instance (optional)
        llm: LLM instance for relevance assessment (optional)
        collection_name: Name of the Qdrant collection
        limit: Maximum number of documents to retrieve
        score_threshold: Minimum similarity score threshold
        
    Returns:
        HybridSearchResult with combined documents
    """
    vector_docs = []
    bm25_docs = []
    
    # Step 1: Vector search
    try:
        query_embedding = embeddings.embed_query(query)
        vector_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=min(limit * 2, 20),  # Get more for better filtering
            score_threshold=max(score_threshold - 0.1, 0.1),  # Slightly lower threshold
            with_payload=True,
            with_vectors=False
        )
        
        for result in vector_results:
            if result.score >= score_threshold:  # Apply final threshold
                # Extract metadata directly from payload since it's not nested
                metadata = {
                    "file_path": result.payload.get("file_path", ""),
                    "created_date": result.payload.get("created_date", ""),
                }
                # Include any additional payload items that aren't chunk or the metadata fields
                for key, value in result.payload.items():
                    if key not in ["chunk", "file_path", "created_date"]:
                        metadata[key] = value
                
                doc = {
                    "id": f"vec_{result.id}",
                    "content": result.payload.get("chunk", ""),
                    "metadata": metadata,
                    "score": float(result.score),
                    "source": "vector_search",
                    "search_type": "vector"
                }
                vector_docs.append(doc)
                
    except Exception as e:
        logger.warning("vector_search_failed", error=str(e))
    
    # Step 2: BM25 keyword search (if available)
    if bm25_retriever:
        try:
            bm25_results = bm25_retriever.get_relevant_documents(query)
            
            for i, doc in enumerate(bm25_results[:limit]):
                # Calculate keyword relevance score
                keyword_score = _calculate_keyword_relevance(query, doc.page_content)
                
                bm25_doc = {
                    "id": f"bm25_{i}",
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": keyword_score,
                    "source": "keyword_search",
                    "search_type": "bm25"
                }
                bm25_docs.append(bm25_doc)
                
        except Exception as e:
            logger.warning("bm25_search_failed", error=str(e))
    
    # Step 3: Merge and deduplicate results with adaptive weighting
    combined_docs = _merge_search_results(vector_docs, bm25_docs, query)
    
    # Step 4: Limit final results
    combined_docs = combined_docs[:limit]
    
    # Step 5: Calculate precision score
    precision_score = _calculate_hybrid_precision(combined_docs, score_threshold)
    
    # Log weighting analysis for performance tracking
    if vector_docs and bm25_docs:
        query_analysis = _analyze_query_for_weighting(query)
        logger.info(
            "hybrid_search_weighting_analysis",
            query=query[:100],
            vector_weight=query_analysis["vector_weight"],
            bm25_weight=query_analysis["bm25_weight"],
            reasoning=query_analysis["reasoning"],
            vector_docs_count=len(vector_docs),
            bm25_docs_count=len(bm25_docs),
            final_docs_count=len(combined_docs),
            precision_score=precision_score
        )
    
    # Determine strategy used
    if vector_docs and bm25_docs:
        strategy = "hybrid"
    elif vector_docs:
        strategy = "vector_only"
    elif bm25_docs:
        strategy = "bm25_only"
    else:
        strategy = "no_results"
    
    return HybridSearchResult(
        documents=combined_docs,
        vector_count=len(vector_docs),
        bm25_count=len(bm25_docs),
        total_found=len(combined_docs),
        search_strategy=strategy,
        precision_score=precision_score
    )


def _merge_search_results(vector_docs: List[Dict], bm25_docs: List[Dict], query: str) -> List[Dict]:
    """
    Merge and deduplicate vector and BM25 results using adaptive equal weighting.
    Adapts weights based on query characteristics for optimal performance.
    """
    merged_docs = []
    seen_content = set()
    
    # Analyze query to determine optimal weighting
    query_analysis = _analyze_query_for_weighting(query)
    vector_weight = query_analysis["vector_weight"]
    bm25_weight = query_analysis["bm25_weight"]
    
    # Helper function to create content signature
    def content_signature(content: str) -> str:
        # Create a signature based on first 100 chars (lowercased, cleaned)
        cleaned = re.sub(r'\W+', ' ', content.lower()).strip()
        return cleaned[:100]
    
    # Normalize vector scores to 0-1 range
    def normalize_vector_score(score: float) -> float:
        # Vector scores are already 0-1, but ensure they're properly bounded
        return max(0.0, min(1.0, score))
    
    # Normalize BM25 scores to 0-1 range using rank-based approach
    def normalize_bm25_score(rank: int, total_docs: int) -> float:
        # Higher rank = lower score (rank 0 = best)
        if total_docs == 0:
            return 0.0
        return max(0.0, 1.0 - (rank / total_docs))
    
    # Process vector docs first
    for doc in vector_docs:
        sig = content_signature(doc["content"])
        if sig not in seen_content:
            # Normalize vector score and apply adaptive weighting
            normalized_vector = normalize_vector_score(doc["score"])
            keyword_bonus = _calculate_keyword_relevance(query, doc["content"])
            
            # Equal weighting with query adaptation
            doc["hybrid_score"] = (
                normalized_vector * vector_weight + 
                keyword_bonus * bm25_weight
            )
            doc["source_primary"] = "vector"
            merged_docs.append(doc)
            seen_content.add(sig)
    
    # Process BM25 docs, avoiding duplicates
    for i, doc in enumerate(bm25_docs):
        sig = content_signature(doc["content"])
        if sig not in seen_content:
            # Normalize BM25 score and apply adaptive weighting
            normalized_bm25 = normalize_bm25_score(i, len(bm25_docs))
            semantic_bonus = _calculate_semantic_relevance(query, doc["content"])
            
            # Equal weighting with query adaptation
            doc["hybrid_score"] = (
                semantic_bonus * vector_weight + 
                normalized_bm25 * bm25_weight
            )
            doc["source_primary"] = "bm25"
            merged_docs.append(doc)
            seen_content.add(sig)
    
    # Sort by hybrid score
    merged_docs.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    return merged_docs


def _analyze_query_for_weighting(query: str) -> Dict[str, Any]:
    """
    Analyze query characteristics to determine optimal vector/BM25 weighting.
    Returns adaptive weights that sum to 1.0 for true equal contribution.
    """
    query_lower = query.lower()
    
    # Count different query characteristics
    exact_match_indicators = len([word for word in query.split() if len(word) > 6])
    technical_terms = len([word for word in query.split() if any(char.isupper() for char in word)])
    question_words = len([word for word in ['what', 'how', 'why', 'when', 'where', 'who'] if word in query_lower])
    
    # Base weights (true equal weighting)
    vector_weight = 0.5
    bm25_weight = 0.5
    
    # Adjust based on query characteristics
    if exact_match_indicators > 2:
        # Queries with many specific terms benefit from keyword search
        bm25_weight += 0.1
        vector_weight -= 0.1
    
    if technical_terms > 1:
        # Technical queries often need exact term matching
        bm25_weight += 0.1
        vector_weight -= 0.1
        
    if question_words > 0:
        # Conceptual questions benefit from semantic search
        vector_weight += 0.1
        bm25_weight -= 0.1
    
    # Ensure weights sum to 1.0
    total = vector_weight + bm25_weight
    vector_weight /= total
    bm25_weight /= total
    
    return {
        "vector_weight": vector_weight,
        "bm25_weight": bm25_weight,
        "reasoning": f"Vector: {vector_weight:.2f}, BM25: {bm25_weight:.2f}"
    }


def _calculate_semantic_relevance(query: str, content: str) -> float:
    """
    Calculate semantic relevance for BM25 documents (simplified approximation).
    This provides a semantic bonus to BM25 results for better hybrid scoring.
    """
    # Simple semantic indicators
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    
    # Calculate overlap
    overlap = len(query_words.intersection(content_words))
    if len(query_words) == 0:
        return 0.0
    
    # Consider synonyms and related terms (simplified)
    semantic_indicators = {
        'accuracy': ['precision', 'correct', 'accurate', 'reliable'],
        'image': ['picture', 'photo', 'visual', 'scan'],
        'concern': ['issue', 'problem', 'worry', 'risk'],
        'label': ['tag', 'annotation', 'classification', 'category']
    }
    
    semantic_bonus = 0.0
    for query_word in query_words:
        if query_word in semantic_indicators:
            related_words = semantic_indicators[query_word]
            if any(word in content_words for word in related_words):
                semantic_bonus += 0.1
    
    base_score = min(overlap / len(query_words), 1.0)
    return min(base_score + semantic_bonus, 1.0)


def _calculate_keyword_relevance(query: str, content: str) -> float:
    """Calculate simple keyword-based relevance score"""
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    
    if not query_words:
        return 0.0
    
    overlap = len(query_words.intersection(content_words))
    return min(overlap / len(query_words), 1.0)


def _calculate_hybrid_precision(documents: List[Dict], threshold: float) -> float:
    """Calculate precision based on hybrid score distribution"""
    if not documents:
        return 0.0
    
    high_quality_docs = [doc for doc in documents if doc.get("hybrid_score", doc.get("score", 0)) >= threshold]
    return len(high_quality_docs) / len(documents)


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
        # Extract metadata directly from payload since it's not nested
        metadata = {
            "file_path": result.payload.get("file_path", ""),
            "created_date": result.payload.get("created_date", ""),
        }
        # Include any additional payload items that aren't chunk or the metadata fields
        for key, value in result.payload.items():
            if key not in ["chunk", "file_path", "created_date"]:
                metadata[key] = value
        
        doc = {
            "id": result.id,
            "content": result.payload.get("chunk", ""),
            "metadata": metadata,
            "score": float(result.score),
            "source": "vector_store"
        }
        documents.append(doc)
    
    # Calculate statistics
    total_found = len(documents)
    
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


class OrchestratorAgent(SecureAgentBase):
    """
    Simplified Function-Calling Orchestrator Agent
    Routes queries using direct tool execution for fast, reliable routing
    Reads: state.query
    Writes: state.route, state.latency_ms["orch"]
    """
    
    def __init__(self, llm: Optional[AzureChatOpenAI] = None):
        super().__init__(AgentRole.ORCHESTRATOR)
        self.llm = llm
        logger.info("orchestrator_agent_initialized", has_llm=self.llm is not None)
    
    def route_query(self, state: WorkflowState) -> WorkflowState:
        """Route the query with medical validation using function calling approach"""
        
        with observability.measure_agent_performance("orch", cast(Dict[str, Any], state)):
            try:
                query = state["query"]
                
                # Step 1: Validate if query is medical/healthcare related
                validation_result = self.invoke_tool("validate_medical_relevance", {
                    "query": query,
                    "llm": self.llm
                })
                
                # Handle non-medical queries immediately
                if not validation_result.is_medical:
                    state["route"] = "none"
                    state["routing_analysis"] = "Non-medical query detected"
                    state["final_answer"] = validation_result.quick_response
                    
                    logger.info(
                        "orchestrator_non_medical_query",
                        query_length=len(query),
                        trace_id=state.get('trace_id')
                    )
                    
                    return state
                
                # Step 2: For medical queries, analyze characteristics and route
                analysis_result = self.invoke_tool("analyze_query_characteristics", {
                    "query": query,
                    "llm": self.llm
                })
                
                # Determine optimal route based on analysis
                routing_result = self.invoke_tool("determine_optimal_route", {"analysis": analysis_result})
                
                # Extract routing information
                route = routing_result.route
                reasoning = routing_result.reasoning
                
                # Create analysis summary
                analysis = f"Intent: {analysis_result.intent}, Entities: {analysis_result.entity_count}, Relationships: {analysis_result.has_relationships}"
                
                # Update state with routing information
                state["route"] = route
                state["routing_analysis"] = analysis
                
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
                errors = state.get("errors") or []
                state["errors"] = errors + [f"Orchestrator function calling error: {str(e)}"]
                state["route"] = "both"  # Safe fallback
                state["routing_analysis"] = "Error during analysis"
                return state


class VectorRAGAgent(SecureAgentBase):
    """
    Function-Calling Hybrid Search Agent
    Performs semantic search with optional BM25 integration and reranking
    Reads: state.query
    Writes: state.vector_docs, state.latency_ms["vec"]
    """
    
    def __init__(self, qdrant_client: QdrantClient, embeddings: Any, 
                 collection_name: str = "documents", llm: Optional[AzureChatOpenAI] = None,
                 bm25_retriever: Optional[BM25Retriever] = None):
        super().__init__(AgentRole.VECTOR_RAG)
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.llm = llm
        self.bm25_retriever = bm25_retriever
    
    def retrieve_documents(self, state: WorkflowState) -> WorkflowState:
        """Retrieve documents using hybrid search function calling approach"""
        
        with observability.measure_agent_performance("vec", cast(Dict[str, Any], state)):
            try:
                query = state["query"]
                
                # Step 1: Perform hybrid search (vector + BM25) using tool
                if self.bm25_retriever:
                    # Use hybrid search when BM25 is available
                    search_result = self.invoke_tool("perform_hybrid_search", {
                        "query": query,
                        "qdrant_client": self.qdrant_client,
                        "embeddings": self.embeddings,
                        "bm25_retriever": self.bm25_retriever,
                        "llm": self.llm,
                        "collection_name": self.collection_name,
                        "limit": 10,
                        "score_threshold": 0.3
                    })
                    
                    documents = search_result.documents
                    search_strategy = search_result.search_strategy
                    vector_count = search_result.vector_count
                    bm25_count = search_result.bm25_count
                    precision_score = search_result.precision_score
                    
                else:
                    # Fallback to vector-only search
                    vector_result = self.invoke_tool("perform_vector_search", {
                        "query": query,
                        "qdrant_client": self.qdrant_client,
                        "embeddings": self.embeddings,
                        "llm": self.llm,
                        "collection_name": self.collection_name,
                        "limit": 10,
                        "score_threshold": 0.3
                    })
                    
                    documents = vector_result.documents
                    search_strategy = "vector_only"
                    vector_count = len(documents)
                    bm25_count = 0
                    precision_score = vector_result.precision_score
                
                # Step 2: Optional additional reranking if LLM available and enough documents
                reranking_applied = False
                if self.llm and len(documents) > 3 and not self.bm25_retriever:
                    # Only do additional reranking if we haven't already done hybrid scoring                    
                    rerank_result = self.invoke_tool("rerank_documents_by_relevance", {

                        "query": query,
                        "documents": documents,
                        "llm": self.llm
                    })
                    documents = rerank_result.documents
                    reranking_applied = rerank_result.reranking_applied
                
                # Update state
                state["vector_docs"] = documents
                
                logger.info(
                    "hybrid_retrieval_function_calling",
                    documents_retrieved=len(documents),
                    search_strategy=search_strategy,
                    vector_count=vector_count,
                    bm25_count=bm25_count,
                    precision_score=precision_score,
                    reranking_applied=reranking_applied,
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("hybrid_rag_function_calling_error", error=str(e), trace_id=state.get('trace_id'))
                errors = state.get("errors") or []
                state["errors"] = errors + [f"Hybrid RAG error: {str(e)}"]
                state["vector_docs"] = []
                return state


class GraphRAGAgent(SecureAgentBase):
    """
    Function-Calling Graph RAG Agent
    Performs entity extraction and graph queries using direct tool execution
    Reads: state.query
    Writes: state.graph_triples, state.latency_ms["graph"]
    """
    
    def __init__(self, neo4j_driver, llm: AzureChatOpenAI):
        super().__init__(AgentRole.GRAPH_RAG)
        self.driver = neo4j_driver
        self.llm = llm
    
    def extract_and_query(self, state: WorkflowState) -> WorkflowState:
        """Extract entities and query knowledge graph using function calling approach"""
        
        with observability.measure_agent_performance("graph", cast(Dict[str, Any], state)):
            try:
                query = state["query"]
                
                # Step 1: Extract entities, relationships, and concepts
                extraction_result = self.invoke_tool("extract_entities_from_query", {
                    "query": query,
                    "llm": self.llm
                })
                
                # Step 2: Execute graph queries based on extraction
                graph_result = self.invoke_tool("execute_graph_queries", {
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
                errors = state.get("errors") or []
                state["errors"] = errors + [f"Graph RAG error: {str(e)}"]
                state["graph_triples"] = []
                return state

# Register all tools with access control
def register_agent_tools():
    """Register all tools with their allowed agent roles"""
    
    # Orchestrator tools
    tool_registry.register_tool(
        validate_medical_relevance,
        ToolMetadata("validate_medical_relevance", [AgentRole.ORCHESTRATOR])
    )
    tool_registry.register_tool(
        analyze_query_characteristics,
        ToolMetadata("analyze_query_characteristics", [AgentRole.ORCHESTRATOR])
    )
    tool_registry.register_tool(
        determine_optimal_route,
        ToolMetadata("determine_optimal_route", [AgentRole.ORCHESTRATOR])
    )
    
    # Vector RAG tools
    tool_registry.register_tool(
        perform_vector_search,
        ToolMetadata("perform_vector_search", [AgentRole.VECTOR_RAG])
    )
    tool_registry.register_tool(
        perform_hybrid_search,
        ToolMetadata("perform_hybrid_search", [AgentRole.VECTOR_RAG])
    )
    tool_registry.register_tool(
        assess_document_relevance,
        ToolMetadata("assess_document_relevance", [AgentRole.VECTOR_RAG])
    )
    tool_registry.register_tool(
        rerank_documents_by_relevance,
        ToolMetadata("rerank_documents_by_relevance", [AgentRole.VECTOR_RAG])
    )
    
    # Graph RAG tools
    tool_registry.register_tool(
        extract_entities_from_query,
        ToolMetadata("extract_entities_from_query", [AgentRole.GRAPH_RAG])
    )
    tool_registry.register_tool(
        execute_graph_queries,
        ToolMetadata("execute_graph_queries", [AgentRole.GRAPH_RAG])
    )

# Initialize tool registry
register_agent_tools()
