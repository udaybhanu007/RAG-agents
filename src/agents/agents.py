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


# Called by: OrchestratorAgent
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
- Medical conditions, diseases, symptoms, treatments, diagnostics
- Anatomy, physiology, medications, procedures, therapies
- Healthcare systems, medical diagnostics, clinical workflows
- Patient care, clinical scenarios, medical consultations
- Medical imaging (X-rays, CT scans, MRI, ultrasound, mammography)
- Medical informatics, health IT, electronic health records (EHR)
- Medical databases, healthcare data management, clinical data
- Medical research, clinical studies, epidemiology
- Healthcare technology, medical devices, telemedicine
- Medical education, clinical training, medical curricula
- Public health, preventive medicine, health policy
- Medical documentation, clinical notes, medical reports
- Hospital management, healthcare administration
- Biomedical engineering in healthcare context
- Medical AI, clinical decision support systems

SPECIFIC MEDICAL CONTEXTS to always classify as MEDICAL:
- Hospital-scale databases, medical data systems
- Chest X-ray databases, medical imaging repositories
- Clinical data collection and management
- Healthcare infrastructure and technology
- Medical dataset construction and analysis
- Patient information systems
- Clinical research databases

NON-MEDICAL queries include:
- General greetings, casual conversation
- Pure technology/programming (without medical context)
- Sports, entertainment, travel, cooking, lifestyle
- Business, finance, general academic topics
- General software development (non-healthcare)
- Non-medical databases or systems

IMPORTANT GUIDELINES:
- If the query mentions medical terms (X-ray, hospital, patient, clinical, diagnostic) → MEDICAL
- If the query is about medical technology or healthcare IT → MEDICAL
- If the query combines medical + technology contexts → MEDICAL
- Medical database construction, medical data analysis → MEDICAL
- Only classify as NON_MEDICAL if completely unrelated to healthcare

Query: "{query}"

Analyze the query carefully for medical context. Respond with only:
MEDICAL or NON_MEDICAL
"""

    try:
        response = llm.invoke(validation_prompt.format(query=query))
        content = str(response.content).strip().upper()
        
        # Enhanced pattern matching with better edge case handling
        if content == "MEDICAL":
            is_medical = True
        elif content == "NON_MEDICAL" or content == "NON-MEDICAL":
            is_medical = False
        elif "MEDICAL" in content and "NON" not in content:
            # Handle cases where LLM returns "MEDICAL" with extra text
            is_medical = True
        elif "NON" in content and "MEDICAL" in content:
            # Handle "NON_MEDICAL" or "NON-MEDICAL" variations
            is_medical = False
        else:
            # Fallback: Check for medical keywords in the query as safety net
            medical_keywords = [
                'medical', 'hospital', 'patient', 'clinical', 'health', 'disease',
                'symptom', 'treatment', 'diagnosis', 'doctor', 'nurse', 'surgery',
                'x-ray', 'ct scan', 'mri', 'ultrasound', 'imaging', 'chest',
                'anatomy', 'physiology', 'medication', 'drug', 'therapeutic',
                'healthcare', 'medicine', 'clinic', 'emergency', 'icu',
                'radiolog', 'patholog', 'cardiolog', 'oncolog', 'neurol'
            ]
            
            query_lower = query.lower()
            has_medical_keywords = any(keyword in query_lower for keyword in medical_keywords)
            
            if has_medical_keywords:
                is_medical = True
                logger.info("medical_validation_fallback_to_keywords", 
                           query=query[:100], llm_response=content[:50])
            else:
                is_medical = False
                logger.warning("medical_validation_unexpected_format", 
                             content=content[:50], query=query[:100])
        
        quick_response = None
        if not is_medical:
            quick_response = "I'm a medical knowledge assistant specialized in healthcare topics. Please ask me questions about medical conditions, treatments, symptoms, medical databases, healthcare technology, or other healthcare-related matters."
        
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


# Called by: OrchestratorAgent
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


# Called by: GraphRAGAgent
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


# Called by: GraphRAGAgent
@tool
def execute_graph_queries(extraction: EntityExtraction, neo4j_driver, original_query: str = "") -> GraphQueryResult:
    """
    Execute adaptive Cypher queries using Vector DB principles: truly dynamic, self-adapting.
    Analyzes query semantically and builds appropriate filters dynamically.
    
    Args:
        extraction: EntityExtraction with entities, relationships, concepts
        neo4j_driver: Neo4j driver instance
        original_query: Original user query for context analysis
        
    Returns:
        GraphQueryResult with retrieved triples
    """
    entities = extraction.entities
    triples = []
    
    with neo4j_driver.session() as session:
        try:
            logger.info("starting_adaptive_graph_query", entities=entities, original_query=original_query[:100])
            
            # Step 1: Dynamic Query Context Analysis (like Vector DB weighting)
            query_context = _analyze_neo4j_query_context(original_query, entities)
            
            # Step 2: Handle specific patient ID queries first
            patient_triples = _handle_patient_id_queries(session, entities)
            if patient_triples:
                triples.extend(patient_triples)
                logger.info("patient_id_query_executed", triples_found=len(patient_triples))
                return GraphQueryResult(triples=triples[:20], queries_executed=1, scenario_used="PATIENT_ID")
            
            # Step 3: Build and execute adaptive query
            if query_context["has_filters"]:
                adaptive_triples = _execute_adaptive_query(session, query_context, original_query, entities)
                triples.extend(adaptive_triples)
                logger.info("adaptive_query_executed", 
                           filters=query_context["demographic_filters"], 
                           ranges=query_context["numerical_ranges"],
                           triples_found=len(adaptive_triples))
            
            # Step 4: Fallback to entity search if no specific patterns
            if not triples:
                fallback_triples = _execute_fallback_entity_search(session, entities)
                triples.extend(fallback_triples)
                logger.info("fallback_search_executed", triples_found=len(fallback_triples))
            
            return GraphQueryResult(
                triples=triples[:20],
                queries_executed=1,
                scenario_used="ADAPTIVE"
            )
                    
        except Exception as e:
            logger.error("adaptive_graph_query_failed", error=str(e))
            return GraphQueryResult(triples=[], queries_executed=0, scenario_used="ERROR")
        
def _analyze_neo4j_query_context(query: str, entities: List[str]) -> Dict[str, Any]:
    """
    Analyze query context like Vector DB approach - dynamic pattern recognition.
    No hardcoded patterns, adapts to query semantics automatically.
    """
    context = {
        "demographic_filters": {},
        "numerical_ranges": {},
        "entity_types": [],
        "query_intent": "general",
        "has_filters": False
    }
    
    query_lower = query.lower()
    
    # Dynamic demographic detection (semantic understanding)
    gender_indicators = {
        "female": "F", "woman": "F", "women": "F", "lady": "F", "ladies": "F",
        "male": "M", "man": "M", "men": "M", "gentleman": "M", "guy": "M"
    }
    for term, value in gender_indicators.items():
        if term in query_lower:
            context["demographic_filters"]["gender"] = value
            context["has_filters"] = True
            break
    
    # Dynamic age detection with semantic understanding
    age_patterns = re.findall(r'(\d+)\s*(?:\+|years?\s+old|years?)', query_lower)
    over_patterns = re.findall(r'(?:over|above|older\s+than|aged\s+over)\s*(\d+)', query_lower)
    under_patterns = re.findall(r'(?:under|below|less\s+than|younger\s+than)\s*(\d+)', query_lower)
    elderly_patterns = ["elderly", "senior", "aged"]
    young_patterns = ["young", "youth", "juvenile"]
    
    if age_patterns or over_patterns or under_patterns:
        if under_patterns:
            age_value = int(under_patterns[0])
            operator = "<"
        elif over_patterns:
            age_value = int(over_patterns[0])
            operator = ">"
        else:
            age_value = int(age_patterns[0])
            operator = ">" if any(word in query_lower for word in ["over", "above", "older", "aged"]) else ">="
        
        context["numerical_ranges"]["age"] = {"value": age_value, "operator": operator}
        context["has_filters"] = True
    elif any(pattern in query_lower for pattern in elderly_patterns):
        # Handle semantic age terms
        context["numerical_ranges"]["age"] = {"value": 65, "operator": ">"}
        context["has_filters"] = True
    elif any(pattern in query_lower for pattern in young_patterns):
        # Handle young patients (typically under 30)
        context["numerical_ranges"]["age"] = {"value": 30, "operator": "<"}
        context["has_filters"] = True
    
    # Dynamic intent detection
    if any(word in query_lower for word in ["most", "common", "frequent", "top", "highest"]):
        context["query_intent"] = "aggregation"
    elif any(word in query_lower for word in ["count", "number", "how many"]):
        context["query_intent"] = "counting"
    elif any(word in query_lower for word in ["multiple"]):
        context["query_intent"] = "multiple_conditions"
    
    return context

def _handle_patient_id_queries(session, entities: List[str]) -> List[str]:
    """Handle specific patient ID queries with exact matching"""
    triples = []
    
    for entity in entities:
        entity_str = str(entity).lower()
        # Only treat as patient ID if explicitly mentioned as "patient" with number
        # Don't treat standalone numbers as patient IDs (could be ages, etc.)
        if "patient" in entity_str and any(char.isdigit() for char in entity_str):
            patient_id = ''.join(filter(str.isdigit, str(entity)))
            if patient_id:
                patient_query = """
                MATCH (p:Patient {id: $patient_id})
                OPTIONAL MATCH (p)-[r1:HAS_FINDING]->(f:Finding)
                OPTIONAL MATCH (p)-[r2:HAS_IMAGE]->(i:Image)
                RETURN p, f, i, r1, r2
                LIMIT 20
                """
                logger.info("executing_patient_query", query=patient_query, patient_id=patient_id)
                result = session.run(patient_query, patient_id=patient_id)
                
                for record in result:
                    if record["p"]:
                        triples.append(f"Patient(id={record['p']['id']}, age_min={record['p'].get('age_min', 'N/A')}, gender={record['p'].get('gender', 'N/A')})")
                    if record["f"]:
                        triples.append(f"Finding(name={record['f']['name']})")
                    if record["r1"]:
                        triples.append(f"Patient-HAS_FINDING->Finding")
                break
    
    return triples

def _execute_adaptive_query(session, context: Dict[str, Any], original_query: str, entities: List[str] = None) -> List[str]:
    """
    Execute adaptive Cypher query based on context analysis.
    Builds query dynamically like Vector DB query construction.
    """
    triples = []
    
    # Base query structure
    match_clause = "MATCH (p:Patient)-[:HAS_FINDING]->(f:Finding)"
    where_conditions = []
    parameters = {}
    
    # Dynamic WHERE clause building
    if "age" in context["numerical_ranges"]:
        age_info = context["numerical_ranges"]["age"]
        where_conditions.append(f"p.age_min {age_info['operator']} $age_threshold")
        parameters["age_threshold"] = age_info["value"]
    
    if "gender" in context["demographic_filters"]:
        where_conditions.append("p.gender = $gender")
        parameters["gender"] = context["demographic_filters"]["gender"]
    
    # Add entity-based filtering for medical conditions
    if entities:
        medical_entities = [e for e in entities if e.lower() not in ['female', 'females', 'male', 'males', 'years old', 'year old', 'patients', 'patient', 'young female patients', 'elderly patients', 'young patients', 'elderly', 'young', 'findings', 'multiple medical conditions', 'medical conditions', 'conditions']]
        if medical_entities:
            # Filter by the first medical entity (most relevant)
            primary_entity = medical_entities[0]
            where_conditions.append("toLower(f.name) CONTAINS $condition")
            parameters["condition"] = primary_entity.lower()
    
    # Build WHERE clause
    where_clause = " AND ".join(where_conditions) if where_conditions else ""
    
    # Build SELECT clause based on intent
    if context["query_intent"] == "aggregation":
        select_clause = "RETURN f.name as finding, count(*) as count ORDER BY count DESC LIMIT 15"
        
        # Build description for results
        desc_parts = []
        if "gender" in context["demographic_filters"]:
            gender_desc = "female" if context["demographic_filters"]["gender"] == "F" else "male"
            desc_parts.append(gender_desc)
        if "age" in context["numerical_ranges"]:
            age_info = context["numerical_ranges"]["age"]
            desc_parts.append(f"patients {age_info['operator']} {age_info['value']}")
        
        description = " ".join(desc_parts) if desc_parts else "patients"
        
    elif context["query_intent"] == "multiple_conditions":
        # Use WITH clause for patient-level aggregation
        select_clause = "WITH p, collect(DISTINCT f.name) as conditions WHERE size(conditions) > 1 RETURN p.id as patient_id, p.age_min as age, p.gender as gender, conditions, size(conditions) as condition_count ORDER BY condition_count DESC LIMIT 20"
        
    else:
        # Default individual finding query
        select_clause = "RETURN p.id as patient_id, p.age_min as age, p.gender as gender, f.name as finding LIMIT 10"
        description = "relationships"
    
    # Build final query
    final_query = match_clause
    if where_clause:
        final_query += f" WHERE {where_clause}"
    final_query += f" {select_clause}"
    
    logger.info("executing_adaptive_query", 
               query=final_query, 
               parameters=parameters,
               context_filters=context["demographic_filters"],
               context_ranges=context["numerical_ranges"])
    
    try:
        result = session.run(final_query, **parameters)
        
        if context["query_intent"] == "aggregation":
            for record in result:
                triples.append(f"Finding({record['finding']}: {record['count']} cases in {description})")
        elif context["query_intent"] == "multiple_conditions":
            for record in result:
                conditions_str = ", ".join(record['conditions'])
                triples.append(f"Patient(id={record['patient_id']}, age={record['age']}, gender={record['gender']}, conditions=[{conditions_str}], count={record['condition_count']})")
        else:
            for record in result:
                # Handle individual finding format  
                triples.append(f"Patient(id={record['patient_id']}, age={record['age']}, gender={record['gender']})-HAS_FINDING->Finding({record['finding']})")
                
    except Exception as e:
        logger.error("adaptive_query_execution_failed", error=str(e), query=final_query)
    
    return triples


def _execute_fallback_entity_search(session, entities: List[str]) -> List[str]:
    """Fallback entity search when no specific patterns match"""
    triples = []
    
    for entity in entities:
        try:
            entity_search_query = """
            MATCH (n)
            WHERE any(prop in keys(n) WHERE toString(n[prop]) CONTAINS $entity)
            RETURN n, labels(n)[0] as node_type
            LIMIT 10
            """
            
            logger.info("executing_fallback_entity_search", entity=str(entity))
            result = session.run(entity_search_query, entity=str(entity))
            
            for record in result:
                node = record["n"]
                node_type = record["node_type"]
                node_id = node.get('id', node.get('name', 'unknown'))
                triples.append(f"{node_type}(id={node_id}, properties={dict(node.items())})")
                        
        except Exception as e:
            logger.debug("fallback_entity_search_failed", entity=entity, error=str(e))
            continue
    
    return triples

# Called by: VectorRAGAgent
@tool
def perform_hybrid_search(query: str, qdrant_client, embeddings, bm25_retriever=None, 
                         llm=None, collection_name: str = "documents", 
                         limit: int = 10, score_threshold: float = 0.3) -> HybridSearchResult:
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
    
    # Step 5: Calculate precision score (simplified)
    precision_score = None
    if combined_docs:
        high_quality_docs = [doc for doc in combined_docs if doc.get("hybrid_score", doc.get("score", 0)) >= score_threshold]
        precision_score = len(high_quality_docs) / len(combined_docs)
    
    # Log hybrid search summary for performance tracking
    if vector_docs and bm25_docs:
        logger.info(
            "hybrid_search_completed",
            query=query[:100],
            vector_docs_count=len(vector_docs),
            bm25_docs_count=len(bm25_docs),
            final_docs_count=len(combined_docs),
            precision_score=precision_score,
            strategy="hybrid"
        )
    elif vector_docs:
        logger.info("hybrid_search_completed", strategy="vector_only", docs_count=len(combined_docs))
    elif bm25_docs:
        logger.info("hybrid_search_completed", strategy="bm25_only", docs_count=len(combined_docs))
    
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
    Merge and deduplicate vector and BM25 results with simple equal weighting.
    """
    merged_docs = []
    seen_content = set()
    
    # Simple equal weighting (0.5 each)
    vector_weight = 0.5
    bm25_weight = 0.5
    
    def content_signature(content: str) -> str:
        """Create content signature for deduplication"""
        return re.sub(r'\W+', ' ', content.lower()).strip()[:100]
    
    # Process vector docs
    for doc in vector_docs:
        sig = content_signature(doc["content"])
        if sig not in seen_content:
            doc["hybrid_score"] = doc["score"] * vector_weight
            doc["source_primary"] = "vector"
            merged_docs.append(doc)
            seen_content.add(sig)
    
    # Process BM25 docs, avoiding duplicates
    for i, doc in enumerate(bm25_docs):
        sig = content_signature(doc["content"])
        if sig not in seen_content:
            # Simple rank-based scoring
            rank_score = max(0.0, 1.0 - (i / max(len(bm25_docs), 1)))
            doc["hybrid_score"] = rank_score * bm25_weight
            doc["source_primary"] = "bm25"
            merged_docs.append(doc)
            seen_content.add(sig)
    
    # Sort by hybrid score
    merged_docs.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return merged_docs


def _calculate_keyword_relevance(query: str, content: str) -> float:
    """Calculate simple keyword overlap score"""
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    
    if not query_words:
        return 0.0
    
    overlap = len(query_words.intersection(content_words))
    return min(overlap / len(query_words), 1.0)

# Called by: VectorRAGAgent
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


# Called by: OrchestratorAgent
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
                # Re-raise the exception since orchestrator should always return a valid route
                raise
    
    def get_workflow_routing(self, state: WorkflowState) -> str:
        """
        Convert orchestrator route decision to workflow routing format.
        This contains simple mapping logic that doesn't require tool governance.
        """
        route = state.get("route", "both")
        
        # Simple mapping logic - no need for tool governance
        if route == "vector":
            return "vector"
        elif route == "graph":
            return "graph"
        elif route == "both":
            return "both_vector_first"  # Start with vector, then graph
        elif route == "none":
            return "none"  # Non-medical query - end workflow
        else:
            # This should never happen since orchestrator always returns valid routes
            logger.error("invalid_route_from_orchestrator", route=route)
            raise ValueError(f"Invalid route received from orchestrator: {route}")
    
    def get_post_vector_routing(self, state: WorkflowState) -> str:
        """
        Determine next step after vector retrieval based on orchestrator's routing decision.
        This contains simple business logic that doesn't require tool governance.
        """
        route = state.get("route", "both")
        
        # Simple business logic - no need for tool governance
        if route == "both":
            return "continue_to_graph"  # Continue with graph retrieval for comprehensive search
        elif route in ["vector", "graph", "none"]:
            return "continue_to_validator"  # Skip graph, go directly to validation
        else:
            # This should never happen since orchestrator always returns valid routes
            logger.error("invalid_route_for_post_vector", route=route)
            raise ValueError(f"Invalid route for post-vector step: {route}")


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
                search_result = self.invoke_tool("perform_hybrid_search", {
                    "query": query,
                    "qdrant_client": self.qdrant_client,
                    "embeddings": self.embeddings,
                    "bm25_retriever": self.bm25_retriever,
                    "llm": self.llm,
                    "collection_name": self.collection_name,
                    "limit": 10                   
                })
                
                documents = search_result.documents
                search_strategy = search_result.search_strategy
                vector_count = search_result.vector_count
                bm25_count = search_result.bm25_count
                precision_score = search_result.precision_score
                
                # Step 2: Optional additional reranking if LLM available and enough documents
                reranking_applied = False
                if self.llm and len(documents) > 3:
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
        perform_hybrid_search,
        ToolMetadata("perform_hybrid_search", [AgentRole.VECTOR_RAG])
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