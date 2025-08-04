"""
Centralized logging configuration for Multi-Agent RAG Workflow

This module provides a consistent logging setup across all components using structlog.
It ensures structured, JSON-formatted logs that are perfect for production monitoring.
"""

import os
import sys
import structlog
from typing import Dict, Any


def configure_logging(
    log_level: str = "INFO",
    enable_json: bool = True,
    enable_colors: bool = False,
    include_trace_id: bool = True
) -> structlog.BoundLogger:
    """
    Configure structlog for the entire application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to output JSON format (recommended for production)
        enable_colors: Whether to colorize console output (useful for development)
        include_trace_id: Whether to include trace IDs in logs
    
    Returns:
        Configured structlog logger instance
    """
    
    # Get log level from environment or parameter
    log_level = os.getenv("LOG_LEVEL", log_level).upper()
    
    # Determine if we're in development mode
    is_development = os.getenv("DEBUG_MODE", "false").lower() == "true"
    enable_colors = enable_colors or is_development
    
    # Configure processors based on environment
    processors = [
        # Filter logs by level
        structlog.stdlib.filter_by_level,
        
        # Add logger name to each log entry
        structlog.stdlib.add_logger_name,
        
        # Add log level to each entry
        structlog.stdlib.add_log_level,
        
        # Format positional arguments
        structlog.stdlib.PositionalArgumentsFormatter(),
        
        # Add timestamp in ISO format
        structlog.processors.TimeStamper(fmt="iso"),
        
        # Add trace information for debugging
        structlog.processors.StackInfoRenderer(),
        
        # Format exception information
        structlog.processors.format_exc_info,
        
        # Handle Unicode characters properly
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add final renderer based on environment
    if enable_json and not is_development:
        # Production: JSON output for log aggregation
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Development: Human-readable console output
        if enable_colors:
            processors.append(structlog.dev.ConsoleRenderer(colors=True))
        else:
            processors.append(structlog.dev.ConsoleRenderer(colors=False))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to work with structlog
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level, logging.INFO)
    )
    
    # Create and return logger
    logger = structlog.get_logger("multi_agent_rag")
    
    logger.info(
        "logging_configured",
        log_level=log_level,
        json_output=enable_json and not is_development,
        colored_output=enable_colors,
        trace_id_enabled=include_trace_id
    )
    
    return logger


def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a logger instance for a specific component
    
    Args:
        name: Optional name for the logger component
    
    Returns:
        structlog.BoundLogger: Configured logger instance
    
    Example:
        # In agents.py
        logger = get_logger("orchestrator_agent")
        logger.info("routing_decision", route="vector", confidence=0.85)
        
        # In validation_synthesis.py  
        logger = get_logger("validator_agent")
        logger.warning("validation_issue", issue="low_confidence", score=0.45)
    """
    if name:
        return structlog.get_logger(name)
    else:
        return structlog.get_logger()


def log_with_context(logger: structlog.BoundLogger, **context) -> structlog.BoundLogger:
    """
    Add context to logger that will be included in all subsequent log entries
    
    Args:
        logger: The logger instance
        **context: Key-value pairs to add as context
    
    Returns:
        Logger with bound context
    
    Example:
        # Add session context that will appear in all logs
        session_logger = log_with_context(
            logger, 
            session_id="session_123", 
            user_id="user_456"
        )
        
        session_logger.info("query_processed", query="What is AI?")
        # Output will include session_id and user_id automatically
    """
    return logger.bind(**context)


# Convenience functions for common logging patterns
def log_agent_start(logger: structlog.BoundLogger, agent_name: str, **kwargs):
    """Log the start of an agent operation"""
    logger.info(
        "agent_operation_started",
        agent=agent_name,
        **kwargs
    )


def log_agent_complete(logger: structlog.BoundLogger, agent_name: str, duration_ms: float, **kwargs):
    """Log the completion of an agent operation"""
    logger.info(
        "agent_operation_completed",
        agent=agent_name,
        duration_ms=duration_ms,
        **kwargs
    )


def log_agent_error(logger: structlog.BoundLogger, agent_name: str, error: str, **kwargs):
    """Log an agent error"""
    logger.error(
        "agent_operation_failed",
        agent=agent_name,
        error=error,
        **kwargs
    )


def log_query_metrics(logger: structlog.BoundLogger, **metrics):
    """Log query processing metrics"""
    logger.info(
        "query_metrics",
        **metrics
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the logging configuration
    print("Testing structlog configuration...")
    
    # Configure logging for development
    logger = configure_logging(
        log_level="DEBUG",
        enable_json=False,
        enable_colors=True
    )
    
    # Test basic logging
    logger.debug("Debug message", component="test")
    logger.info("Info message", status="testing", count=42)
    logger.warning("Warning message", issue="minor")
    logger.error("Error message", error_code=500)
    
    # Test context binding
    session_logger = log_with_context(logger, session_id="test_123", trace_id="trace_456")
    session_logger.info("Context test", action="processing")
    
    # Test convenience functions
    log_agent_start(logger, "test_agent", query="test query")
    log_agent_complete(logger, "test_agent", 123.45, status="success")
    log_agent_error(logger, "test_agent", "Test error message", error_code="TEST001")
    
    print("✅ Logging configuration test completed!")
