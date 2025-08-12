import re
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SecurityViolationError(Exception):
    """Raised when a security validation fails"""
    pass

class QuerySecurityValidator:
    """
    Simple security validator for user queries
    """
    
    def __init__(self):
        # Basic configuration
        self.max_length = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
        
        # Common malicious patterns
        self.bad_patterns = [
            r"(?i)(union\s+select|drop\s+table|delete\s+from)",
            r"(?i)(<script|javascript:|alert\s*\()",
            r"(?i)(system\s*\(|exec\s*\(|shell_exec)",
            r"(\.\.\/|\.\.\\)",
            r"(?i)(\$where|\$regex)",
        ]
        self.compiled_patterns = [re.compile(p) for p in self.bad_patterns]
    
    def validate_query(self, query: str) -> str:
        """
        Simple query validation and sanitization
        
        Args:
            query: The user query to validate
            
        Returns:
            str: Sanitized and validated query
            
        Raises:
            SecurityViolationError: If validation fails
        """
        # Check length
        if len(query) > self.max_length:
            raise SecurityViolationError(f"Query too long: {len(query)} chars (max: {self.max_length})")
        
        if len(query.strip()) == 0:
            raise SecurityViolationError("Empty query not allowed")
        
        # Check for malicious patterns
        for pattern in self.compiled_patterns:
            if pattern.search(query):
                raise SecurityViolationError("Potentially malicious pattern detected")
        
        # Clean the query
        sanitized = self._clean_query(query)
        
        logger.info(f"Query validated: {len(query)} -> {len(sanitized)} chars")
        return sanitized
    
    def _clean_query(self, query: str) -> str:
        """Clean and sanitize the query with minimal changes"""
        # Remove control characters only
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)
        
        # Only replace truly dangerous characters, not normal query characters
        replacements = {
            '<script': '&lt;script',  # Only replace script tags
            '</script': '&lt;/script',
            'javascript:': 'javascript&#58;',
            '--': '&#45;&#45;',  # SQL comment
            '/*': '&#47;&#42;',  # SQL comment start
            '*/': '&#42;&#47;',  # SQL comment end
        }
        
        for dangerous, safe in replacements.items():
            cleaned = cleaned.replace(dangerous.lower(), safe)
            cleaned = cleaned.replace(dangerous.upper(), safe)
        
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
