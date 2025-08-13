"""
Input sanitization and prompt injection protection utilities

This module provides security functions for detecting and preventing
prompt injection attacks in LLM interactions.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def detect_prompt_injection(user_input: str) -> bool:
    """
    Detect potential prompt injection attempts
    
    Args:
        user_input: The user input to analyze
        
    Returns:
        bool: True if injection detected, False otherwise
    """
    # Normalize input for analysis
    normalized = user_input.lower().strip()
    
    # Comprehensive prompt injection patterns
    injection_patterns = [
        # Direct prompt manipulation
        r'ignore\s+(previous|above|all)\s+(instructions|commands|prompts)',
        r'forget\s+(previous|above|all)\s+(instructions|commands|prompts)',
        r'disregard\s+(previous|above|all)\s+(instructions|commands|prompts)',
        
        # Role manipulation
        r'you\s+are\s+(now|a|an)\s+',
        r'act\s+as\s+(a|an)\s+',
        r'pretend\s+to\s+be\s+',
        r'roleplay\s+as\s+',
        
        # System instruction overrides
        r'system\s*:\s*',
        r'assistant\s*:\s*',
        r'human\s*:\s*',
        r'user\s*:\s*',
        
        # Template injection
        r'\{\{\s*.*\s*\}\}',
        r'\$\{.*\}',
        r'<%.*%>',
        
        # Direct instruction injection
        r'respond\s+with\s+only',
        r'output\s+only',
        r'say\s+only',
        r'print\s+only',
        
        # Context breaking
        r'end\s+of\s+prompt',
        r'new\s+prompt\s*:',
        r'updated\s+instructions\s*:',
        
        # Advanced injection patterns
        r'\\n\\n.*system',
        r'---\s*new\s+instructions',
        r'\\[INST\\]',
        r'<\\|.*\\|>',
    ]
    
    # Check for injection patterns
    for pattern in injection_patterns:
        if re.search(pattern, normalized):
            logger.warning(f"prompt_injection_detected: {pattern} in input: {user_input[:50]}")
            return True
    
    return False


def sanitize_user_input(user_input: str) -> str:
    """
    Sanitize user input to prevent injection
    
    Args:
        user_input: Raw user input
        
    Returns:
        str: Sanitized input
    """
    # Remove potential injection markers
    sanitized = re.sub(r'[{}$<>\\[\\]]', '', user_input)
    
    # Remove common injection delimiters
    sanitized = re.sub(r'\\n\\n', ' ', sanitized)
    sanitized = re.sub(r'---+', ' ', sanitized)
    
    # Limit length to prevent overflow attacks
    max_length = 2000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.info(f"input_truncated: {len(user_input)} -> {len(sanitized)}")
    
    return sanitized.strip()


def validate_llm_output(output: str) -> str:
    """
    Validate and filter LLM output for security
    
    Args:
        output: Raw LLM output
        
    Returns:
        str: Validated output
    """
    # Check for potential data exfiltration attempts
    suspicious_patterns = [
        r'system\s+prompt\s*:',
        r'internal\s+instructions\s*:',
        r'debug\s+mode\s*:',
        r'configuration\s*:',
        r'api\s+key',
        r'secret\s+key',
        r'password',
        r'token\s*:',
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, output.lower()):
            logger.warning(f"suspicious_llm_output_detected: {pattern}")
            return "suspicious LLM output detected."
    
    return output


def create_secure_prompt_template(template: str, user_input: str, **kwargs) -> str:
    """
    Create a secure prompt using parameterized templates and input delimiters
    
    Args:
        template: The prompt template with placeholders
        user_input: The user input to sanitize and include
        **kwargs: Additional template parameters
        
    Returns:
        str: Formatted secure prompt
    """
    # Step 1: Detect injection attempts
    # if detect_prompt_injection(user_input):
    #     logger.warning(f"prompt_injection_blocked_in_template: {user_input[:50]}")
    #     raise ValueError("Prompt injection attempt detected")
    
    # Step 2: Sanitize user input
    sanitized_input = sanitize_user_input(user_input)
    
    # Step 3: Format template with sanitized input and delimiters
    template_params = {
        'user_query': sanitized_input,
        'sanitized_query': sanitized_input,  # Add this for templates that use sanitized_query
        **kwargs
    }
    
    try:
        formatted_prompt = template.format(**template_params)
        return formatted_prompt
    except KeyError as e:
        logger.error(f"template_formatting_error: missing key {e}, template: {template[:100]}")
        raise ValueError(f"Template formatting error: missing key {e}")


def secure_llm_interaction(llm, template: str, user_input: str, **kwargs) -> str:
    """
    Perform a secure LLM interaction with full protection pipeline
    
    Args:
        llm: The LLM instance
        template: The prompt template
        user_input: The user input
        **kwargs: Additional template parameters
        
    Returns:
        str: Validated LLM response
    """
    try:
        # Create secure prompt
        formatted_prompt = create_secure_prompt_template(template, user_input, **kwargs)
        
        # Get LLM response
        response = llm.invoke(formatted_prompt)
        
        # Validate output
        raw_content = str(response.content).strip()
        validated_content = validate_llm_output(raw_content)
        
        return validated_content
        
    except ValueError as e:
        # Handle injection attempts gracefully
        logger.warning(f"secure_llm_interaction_blocked: {str(e)}")
        return "LLM response is not secured"
    except Exception as e:
        logger.error(f"secure_llm_interaction_failed: {str(e)}")
        raise


# Secure prompt templates with input delimiters
MEDICAL_VALIDATION_TEMPLATE = """
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

<USER_QUERY>{user_query}</USER_QUERY>

Analyze the query carefully for medical context. Respond with only:
MEDICAL or NON_MEDICAL
"""

QUERY_ANALYSIS_TEMPLATE = """
Analyze this medical query and classify its characteristics:

INTENT TYPES:
- FACTUAL: Seeks specific facts, definitions, symptoms, procedures
- RELATIONAL: Explores connections, relationships, interactions between entities
- ANALYTICAL: Requires comparison, evaluation, analysis of multiple aspects

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

<USER_QUERY>{user_query}</USER_QUERY>

Respond in this exact format:
INTENT: [FACTUAL|RELATIONAL|ANALYTICAL]
ENTITY_COUNT: [1|2|3]
HAS_RELATIONSHIPS: [true|false]
REASONING: [Brief explanation focusing on entities and relationships detected]
"""

ENTITY_EXTRACTION_TEMPLATE = """
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

<USER_QUERY>{user_query}</USER_QUERY>

OUTPUT FORMAT (REQUIRED):
ENTITIES: [entity1, entity2, entity3]
RELATIONSHIPS: [relationship1, relationship2]
CONCEPTS: [concept1, concept2]
"""

DOCUMENT_RERANKING_TEMPLATE = """
<USER_QUERY>{user_query}</USER_QUERY>

Rank these documents by relevance (1 = most relevant):

{document_list}

Return only numbers separated by commas (e.g., 3,1,4,2):
"""

SYNTHESIS_TEMPLATE = """
Answer this query using the provided information sources.

Query: 
<USER_QUERY>{sanitized_query}</USER_QUERY>

Available Information:
{vector_content}

{graph_content}

Instructions:
1. Provide a comprehensive answer using the available information
2. Be factual and accurate
3. If sources are limited, mention this limitation

Answer:
"""
