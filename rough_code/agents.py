import os
from typing import List, Dict, Any, Optional
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
import numpy as np
from workflow_state import WorkflowState
from observability import observability
from logging_config import get_logger

logger = get_logger("agents")


class OrchestratorAgent:
    """
    Orchestrator Agent - Routes queries to appropriate retrieval agents
    Reads: state.query
    Writes: state.route, state.latency_ms["orch"]
    """
    
    def __init__(self, llm: AzureChatOpenAI):
        self.llm = llm
        self.routing_prompt = """
        You are a query routing expert. Analyze the query using a systematic two-step approach to determine the optimal retrieval strategy.

        STEP 1: QUERY ANALYSIS
        First, analyze the query for these specific characteristics:
        
        Query: {query}
        
        A. Intent Classification:
        - FACTUAL: Seeking specific facts, definitions, or descriptions
        - RELATIONAL: Exploring connections, relationships, or dependencies between entities
        - ANALYTICAL: Requiring comparison, synthesis, or complex reasoning
        - PROCEDURAL: Looking for step-by-step processes or workflows
        
        B. Entity Detection:
        - Are there 2+ named entities that might be connected?
        - Does the query ask about relationships using words like: "between", "connected to", "related", "linked", "associated with", "depends on"?
        
        C. Scope Assessment:
        - NARROW: Single concept or entity
        - MEDIUM: Multiple related concepts
        - BROAD: Complex multi-faceted question
        
        STEP 2: ROUTING DECISION
        Based on your analysis, select ONE route using these precise rules:
        
        ROUTE: "vector"
        Use when:
        ✓ FACTUAL intent + single/few entities
        ✓ Semantic similarity needed ("similar to", "like", "about")
        ✓ Document content retrieval
        ✓ Definition or description requests
        
        Examples:
        - "What is pneumonia?"
        - "Find documents about chest X-ray analysis"
        - "Explain cardiomegaly findings"
        - "Show me content similar to pulmonary edema"
        - "Define atelectasis in medical imaging"
        - "What are the symptoms of pleural effusion?"
        
        ROUTE: "graph"
        Use when:
        ✓ RELATIONAL intent + multiple entities
        ✓ Explicit relationship queries ("how X relates to Y")
        ✓ Network/connection exploration
        ✓ Dependency or hierarchy questions
        
        Examples:
        - "How is pneumonia connected to lung opacity?"
        - "What is the relationship between cardiomegaly and heart failure?"
        - "Show connections between chest X-ray findings and patient symptoms"
        - "How do different pathologies depend on each other?"
        - "What links pneumothorax to respiratory distress?"
        - "How are consolidation patterns related to infection types?"
        
        ROUTE: "both"
        Use when:
        ✓ ANALYTICAL intent + complex scope
        ✓ Requires both semantic content AND relationship data
        ✓ Comparison queries involving multiple entities
        ✓ Comprehensive analysis needed
        
        Examples:
        - "Compare pneumonia and pneumothorax findings and their diagnostic relationships"
        - "Analyze the clinical presentation of various chest pathologies and their interconnections"
        - "How do different imaging techniques relate to chest X-ray diagnosis, and what are their diagnostic strengths?"
        - "Compare cardiomegaly and pleural effusion manifestations and their clinical correlations"
        
        ROUTE: "none"
        Use when:
        ✓ Query is outside available domain knowledge
        ✓ Requires real-time data not in the system
        ✓ Personal opinions or subjective judgments
        ✓ System/meta questions about the AI itself
        
        Examples:
        - "What's the weather today?"
        - "What do you think about this?"
        - "How are you feeling?"
        - "What's happening in the news?"
        
        CRITICAL RULES:
        1. If query contains relationship indicators ("between", "connected", "linked", "related to") + 2+ entities → ALWAYS consider "graph" or "both"
        2. If query is purely factual about single entity → PREFER "vector"
        3. If unsure between two routes → DEFAULT to "both"
        4. Never guess or assume domain knowledge not explicitly mentioned
        
        OUTPUT FORMAT (REQUIRED):
        ANALYSIS: [Brief analysis of intent, entities, and scope]
        ROUTE: [vector|graph|both|none]
        REASON: [Specific justification referencing the rules above]
        CONFIDENCE: [HIGH|MEDIUM|LOW]
        """
    
    def route_query(self, state: WorkflowState) -> WorkflowState:
        """Route the query to appropriate retrieval agents"""
        
        with observability.measure_agent_performance("orch", state):
            try:
                # Log routing start
                observability.log_query_start(state)
                
                query = state["query"]
                
                # Use LLM to make routing decision
                prompt = self.routing_prompt.format(query=query)
                response = self.llm.invoke(prompt)
                
                # Parse response
                content = response.content.strip()
                
                # Extract different components
                analysis_line = [line for line in content.split('\n') if line.startswith('ANALYSIS:')]
                route_line = [line for line in content.split('\n') if line.startswith('ROUTE:')]
                reason_line = [line for line in content.split('\n') if line.startswith('REASON:')]
                confidence_line = [line for line in content.split('\n') if line.startswith('CONFIDENCE:')]
                
                if route_line:
                    route = route_line[0].split('ROUTE:')[1].strip().lower()
                    reason = reason_line[0].split('REASON:')[1].strip() if reason_line else "No reason provided"
                    analysis = analysis_line[0].split('ANALYSIS:')[1].strip() if analysis_line else "No analysis provided"
                    confidence = confidence_line[0].split('CONFIDENCE:')[1].strip() if confidence_line else "MEDIUM"
                else:
                    # Simple fallback - default to 'both' for safety
                    route = "both"
                    reason = "LLM response parsing failed - defaulting to comprehensive retrieval"
                    analysis = "Unable to parse LLM analysis"
                    confidence = "LOW"
                    
                    # Log the parsing failure for debugging
                    logger.warning(
                        "llm_response_parsing_failed",
                        llm_response=content[:200] + "..." if len(content) > 200 else content,
                        trace_id=state.get('trace_id')
                    )
                
                # Validate route
                valid_routes = ["vector", "graph", "both", "none"]
                if route not in valid_routes:
                    original_route = route
                    route = "both"  # Default fallback
                    reason = f"Invalid route detected, defaulting to 'both'. Original: {original_route}"
                    confidence = "LOW"
                
                # Update state with enhanced routing information
                state["route"] = route
                state["routing_analysis"] = analysis
                state["routing_confidence"] = confidence
                
                # Log routing decision
                observability.log_routing_decision(state, reason)
                
                logger.info(
                    "orchestrator_routing",
                    route=route,
                    reason=reason,
                    analysis=analysis,
                    confidence=confidence,
                    query_length=len(query),
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("orchestrator_error", error=str(e), trace_id=state.get('trace_id'))
                state["errors"] = state.get("errors", []) + [f"Orchestrator error: {str(e)}"]
                state["route"] = "both"  # Safe fallback
                return state


class VectorRAGAgent:
    """
    Vector-RAG Agent - Performs semantic search using Qdrant
    Reads: state.query
    Tool: Qdrant search
    Optional: LLM re-rank
    Writes: state.vector_docs, state.latency_ms["vec"], state.memory_usage["vec"]
    """
    
    def __init__(self, qdrant_client: QdrantClient, embeddings: AzureOpenAIEmbeddings, 
                 collection_name: str = "documents", llm: AzureChatOpenAI = None):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.llm = llm  # For optional re-ranking
    
    def retrieve_documents(self, state: WorkflowState) -> WorkflowState:
        """
        Retrieve documents using vector similarity search
        """
        
        with observability.measure_agent_performance("vec", state):
            try:
                query = state["query"]
                
                # Standard search parameters
                limit = 10
                score_threshold = 0.6
                
                # Generate query embedding
                query_embedding = self.embeddings.embed_query(query)
                
                # Search in Qdrant
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
                
                # Convert results to standard format
                vector_docs = []
                for result in search_results:
                    doc = {
                        "id": result.id,
                        "content": result.payload.get("content", ""),
                        "metadata": result.payload.get("metadata", {}),
                        "score": float(result.score),
                        "source": "vector_store"
                    }
                    vector_docs.append(doc)
                
                # Optional re-ranking if LLM is available
                if self.llm and len(vector_docs) > 3:
                    vector_docs = self._rerank_documents(query, vector_docs)
                
                state["vector_docs"] = vector_docs
                
                # Log retrieval results
                observability.log_retrieval_results(state, "vector")
                
                logger.info(
                    "vector_retrieval_completed",
                    documents_retrieved=len(vector_docs),
                    avg_score=sum(doc["score"] for doc in vector_docs) / len(vector_docs) if vector_docs else 0,
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("vector_rag_error", error=str(e), trace_id=state.get('trace_id'))
                state["errors"] = state.get("errors", []) + [f"Vector RAG error: {str(e)}"]
                if not state.get("vector_docs"):
                    state["vector_docs"] = []
                return state
    
    def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM to re-rank documents for better relevance"""
        try:
            rerank_prompt = f"""
            Query: {query}
            
            Rank the following documents by relevance to the query (1 = most relevant):
            
            {chr(10).join([f"{i+1}. {doc['content'][:200]}..." for i, doc in enumerate(documents)])}
            
            Return only the ranking as numbers separated by commas (e.g., 3,1,4,2):
            """
            
            response = self.llm.invoke(rerank_prompt)
            rankings = [int(x.strip()) - 1 for x in response.content.strip().split(',')]
            
            # Reorder documents based on rankings
            if len(rankings) == len(documents):
                return [documents[i] for i in rankings if 0 <= i < len(documents)]
            
        except Exception as e:
            logger.warning("reranking_failed", error=str(e))
        
        return documents  # Return original order if reranking fails


class GraphRAGAgent:
    """
    Graph-RAG Agent - Performs entity extraction and graph queries using Neo4j
    Reads: state.query
    Tool: Neo4j Cypher
    Entity extraction
    Writes: state.graph_triples, state.latency_ms["graph"], state.memory_usage["graph"]
    """
    
    def __init__(self, neo4j_driver, llm: AzureChatOpenAI):
        self.driver = neo4j_driver
        self.llm = llm
        self.entity_extraction_prompt = """
        You are an expert entity extraction specialist for medical knowledge graphs. Extract named entities, relationships, and concepts from the query using a systematic approach.

        STEP 1: ENTITY IDENTIFICATION
        Identify these entity types:
        - MEDICAL CONDITIONS: diseases, disorders, pathologies (e.g., pneumonia, cardiomegaly)
        - ANATOMICAL STRUCTURES: body parts, organs (e.g., lung, heart, chest)
        - MEDICAL PROCEDURES: tests, imaging, treatments (e.g., chest X-ray, CT scan)
        - CLINICAL FINDINGS: symptoms, signs (e.g., dyspnea, opacity, consolidation)
        - CONTEXTUAL: severity, location, timing (e.g., acute, bilateral, upper lobe)

        STEP 2: RELATIONSHIP IDENTIFICATION
        Identify relationship indicators:
        - CAUSATIVE: "causes", "leads to", "results in"
        - ASSOCIATED: "associated with", "related to", "linked to"
        - DIAGNOSTIC: "indicates", "suggests", "shows"
        - LOCATIONAL: "located in", "affects", "involves"

        STEP 3: CONCEPT EXTRACTION
        Extract broader medical domains and specialties.

        EXAMPLES:

        Query: "What is the relationship between pneumonia and lung opacity in chest X-rays?"
        ENTITIES: [pneumonia, lung opacity, chest X-rays, lung]
        RELATIONSHIPS: [relationship between, shows, indicates]
        CONCEPTS: [respiratory diseases, diagnostic imaging, radiological findings]

        Query: "How does cardiomegaly affect cardiac function?"
        ENTITIES: [cardiomegaly, cardiac function, heart]
        RELATIONSHIPS: [affects, causes, impacts]
        CONCEPTS: [cardiac disorders, cardiovascular pathology]

        Query: "Compare pleural effusion symptoms with pneumothorax"
        ENTITIES: [pleural effusion, pneumothorax, symptoms]
        RELATIONSHIPS: [compare, similar to, different from]
        CONCEPTS: [pleural diseases, respiratory symptoms]

        EXTRACTION RULES:
        1. Extract 3-6 entities maximum for precision
        2. Focus on medically relevant terms only
        3. Normalize medical terms when possible
        4. Capture relationship words exactly as they appear

        OUTPUT FORMAT (REQUIRED):
        ENTITIES: [entity1, entity2, entity3]
        RELATIONSHIPS: [relationship1, relationship2]
        CONCEPTS: [concept1, concept2]
        """
    
    def extract_and_query(self, state: WorkflowState) -> WorkflowState:
        """
        Extract entities and query the knowledge graph
        """
        
        with observability.measure_agent_performance("graph", state):
            try:
                query = state["query"]
                
                # Extract entities using LLM
                entities_response = self.llm.invoke(
                    self.entity_extraction_prompt.format(query=query)
                )
                
                # Parse entity extraction results
                entities, relationships, concepts = self._parse_entity_response(entities_response.content)
                
                # Build Cypher queries
                cypher_queries = self._build_cypher_queries(entities, relationships, concepts)
                
                # Log query building strategy
                scenario = self._identify_query_scenario(entities, relationships, concepts)
                logger.info(
                    "graph_query_scenario_selected",
                    scenario=scenario,
                    entities_count=len(entities),
                    relationships_count=len(relationships),
                    concepts_count=len(concepts),
                    queries_generated=len(cypher_queries),
                    trace_id=state.get('trace_id')
                )
                
                # Execute queries and collect results
                graph_triples = []
                
                with self.driver.session() as session:
                    for cypher_query in cypher_queries:
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
                                graph_triples.append(triple)
                        except Exception as e:
                            logger.warning("cypher_query_failed", query=cypher_query, error=str(e))
                
                state["graph_triples"] = graph_triples
                
                # Log retrieval results
                observability.log_retrieval_results(state, "graph")
                
                logger.info(
                    "graph_retrieval_completed",
                    entities_found=len(entities),
                    relationships_found=len(relationships),
                    triples_retrieved=len(graph_triples),
                    trace_id=state.get('trace_id')
                )
                
                return state
                
            except Exception as e:
                logger.error("graph_rag_error", error=str(e), trace_id=state.get('trace_id'))
                state["errors"] = state.get("errors", []) + [f"Graph RAG error: {str(e)}"]
                if not state.get("graph_triples"):
                    state["graph_triples"] = []
                return state
    
    def _parse_entity_response(self, response: str) -> tuple:
        """Parse LLM response for entity extraction"""
        entities, relationships, concepts = [], [], []
        
        try:
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('ENTITIES:'):
                    entities_text = line.split('ENTITIES:')[1].strip()
                    if entities_text and entities_text != '[]':
                        entities = [e.strip() for e in entities_text.strip('[]').split(',')]
                elif line.startswith('RELATIONSHIPS:'):
                    rel_text = line.split('RELATIONSHIPS:')[1].strip()
                    if rel_text and rel_text != '[]':
                        relationships = [r.strip() for r in rel_text.strip('[]').split(',')]
                elif line.startswith('CONCEPTS:'):
                    concepts_text = line.split('CONCEPTS:')[1].strip()
                    if concepts_text and concepts_text != '[]':
                        concepts = [c.strip() for c in concepts_text.strip('[]').split(',')]
        except Exception as e:
            logger.warning("entity_parsing_failed", error=str(e))
        
        return entities, relationships, concepts
    
    def _identify_query_scenario(self, entities: List[str], relationships: List[str], concepts: List[str]) -> str:
        """Identify which query scenario should be used"""
        if len(entities) >= 2 and relationships:
            return "MULTI_ENTITY_WITH_RELATIONSHIPS"
        elif len(entities) >= 2 and not relationships:
            return "MULTI_ENTITY_NO_RELATIONSHIPS"
        elif len(entities) == 1 and relationships:
            return "SINGLE_ENTITY_WITH_RELATIONSHIPS"
        elif len(entities) == 1 and not relationships:
            return "SINGLE_ENTITY_NO_RELATIONSHIPS"
        elif not entities and not relationships and concepts:
            return "CONCEPTS_ONLY"
        else:
            # This should rarely happen with a well-designed extraction prompt
            return "CONCEPTS_ONLY"  # Default to concept-based search as most reliable fallback
    
    
    def _build_cypher_queries(self, entities: List[str], relationships: List[str], 
                            concepts: List[str]) -> List[str]:
        """Build Cypher queries based on scenario identification"""
        queries = []
        
        # Get scenario using single source of truth
        scenario = self._identify_query_scenario(entities, relationships, concepts)
        
        # Execute queries based on identified scenario
        if scenario == "MULTI_ENTITY_WITH_RELATIONSHIPS":
            queries = self._build_multi_entity_relationship_queries(entities, relationships)
            
        elif scenario == "MULTI_ENTITY_NO_RELATIONSHIPS":
            queries = self._build_multi_entity_queries(entities)
            
        elif scenario == "SINGLE_ENTITY_WITH_RELATIONSHIPS":
            queries = self._build_single_entity_relationship_queries(entities[0], relationships)
            
        elif scenario == "SINGLE_ENTITY_NO_RELATIONSHIPS":
            queries = self._build_single_entity_queries(entities[0])
            
        elif scenario == "CONCEPTS_ONLY":
            queries = self._build_concept_queries(concepts)
        
        # Add concept enhancement for scenarios with concepts (supplementary)
        if concepts and queries and scenario != "CONCEPTS_ONLY":
            concept_queries = self._build_concept_enhancement_queries(concepts)
            queries.extend(concept_queries)
        
        return queries
    
    def _build_multi_entity_relationship_queries(self, entities: List[str], relationships: List[str]) -> List[str]:
        """Build queries for multiple entities with explicit relationships"""
        queries = []
        
        # Entity-to-entity relationships
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                query = f"""
                MATCH (a)-[r]-(b) 
                WHERE (toLower(a.name) CONTAINS toLower('{entities[i]}') 
                       OR toLower(a.title) CONTAINS toLower('{entities[i]}'))
                AND (toLower(b.name) CONTAINS toLower('{entities[j]}') 
                     OR toLower(b.title) CONTAINS toLower('{entities[j]}'))
                RETURN a.name as subject, type(r) as predicate, b.name as object
                LIMIT 5
                """
                queries.append(query)
        
        # Relationship-specific searches with primary entity
        primary_entity = entities[0]
        for relationship in relationships:
            query = f"""
            MATCH (a)-[r]-(b)
            WHERE (toLower(a.name) CONTAINS toLower('{primary_entity}')
                   OR toLower(a.title) CONTAINS toLower('{primary_entity}'))
            AND (toLower(type(r)) CONTAINS toLower('{relationship}')
                 OR toLower(r.type) CONTAINS toLower('{relationship}'))
            RETURN a.name as subject, type(r) as predicate, b.name as object
            LIMIT 5
            """
            queries.append(query)
        
        return queries
    
    def _build_multi_entity_queries(self, entities: List[str]) -> List[str]:
        """Build queries for multiple entities without explicit relationships"""
        queries = []
        
        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                query = f"""
                MATCH (a)-[r]-(b) 
                WHERE (toLower(a.name) CONTAINS toLower('{entities[i]}') 
                       OR toLower(a.title) CONTAINS toLower('{entities[i]}'))
                AND (toLower(b.name) CONTAINS toLower('{entities[j]}') 
                     OR toLower(b.title) CONTAINS toLower('{entities[j]}'))
                RETURN a.name as subject, type(r) as predicate, b.name as object
                LIMIT 8
                """
                queries.append(query)
        
        return queries
    
    def _build_single_entity_relationship_queries(self, entity: str, relationships: List[str]) -> List[str]:
        """Build queries for single entity with explicit relationships"""
        queries = []
        
        for relationship in relationships:
            query = f"""
            MATCH (a)-[r]-(b)
            WHERE (toLower(a.name) CONTAINS toLower('{entity}')
                   OR toLower(a.title) CONTAINS toLower('{entity}'))
            AND (toLower(type(r)) CONTAINS toLower('{relationship}')
                 OR toLower(r.type) CONTAINS toLower('{relationship}'))
            RETURN a.name as subject, type(r) as predicate, b.name as object
            LIMIT 8
            """
            queries.append(query)
        
        return queries
    
    def _build_single_entity_queries(self, entity: str) -> List[str]:
        """Build queries for single entity without relationships"""
        queries = []
        
        # Direct entity properties
        query = f"""
        MATCH (n) 
        WHERE toLower(n.name) CONTAINS toLower('{entity}') 
        OR toLower(n.title) CONTAINS toLower('{entity}')
        RETURN n.name as subject, 'has_property' as predicate, n as object
        LIMIT 5
        """
        queries.append(query)
        
        # Entity connections
        query = f"""
        MATCH (a)-[r]-(b)
        WHERE toLower(a.name) CONTAINS toLower('{entity}')
        OR toLower(a.title) CONTAINS toLower('{entity}')
        RETURN a.name as subject, type(r) as predicate, b.name as object
        LIMIT 8
        """
        queries.append(query)
        
        return queries
    
    def _build_concept_queries(self, concepts: List[str]) -> List[str]:
        """Build queries for concept-only scenarios"""
        queries = []
        
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
        
        return queries
    
    def _build_concept_enhancement_queries(self, concepts: List[str]) -> List[str]:
        """Build supplementary concept queries for enhancement"""
        queries = []
        
        # Limit to 1-2 most relevant concepts to avoid query explosion
        for concept in concepts[:2]:
            query = f"""
            MATCH (n) 
            WHERE toLower(n.category) CONTAINS toLower('{concept}')
            OR toLower(n.domain) CONTAINS toLower('{concept}')
            RETURN n.name as subject, 'belongs_to_concept' as predicate, '{concept}' as object
            LIMIT 3
            """
            queries.append(query)
        
        return queries
