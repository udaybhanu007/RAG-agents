import logging
from .security_validator import QuerySecurityValidator, SecurityViolationError

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    """
    Simple security middleware for RAG workflow
    Handles security validation
    """
    
    def __init__(self):
        self.validator = QuerySecurityValidator()
    
    def validate_and_sanitize_query(self, query: str) -> str:
        """
        Validate and sanitize query
        
        Args:
            query: User query to validate
            
        Returns:
            str: Sanitized query
            
        Raises:
            SecurityViolationError: If validation fails
        """
        # Validate query
        sanitized_query = self.validator.validate_query(query)
        
        logger.info(f"Query validated: {len(query)} -> {len(sanitized_query)} chars")
        return sanitized_query
