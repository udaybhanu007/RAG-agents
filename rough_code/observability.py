"""
Simplified observability module for LangSmith integration

This module provides performance tracking and structured logging for multi-agent workflows.
Prometheus metrics have been removed in favor of LangSmith tracing and structured logging.
"""

import time
import os
from typing import Dict, Any, Optional
from contextlib import contextmanager
from logging_config import get_logger

# Try to import psutil for memory tracking (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Structured logging for LangSmith integration
logger = get_logger("observability")


class ObservabilityManager:
    """Simplified observability for LangSmith integration"""
    
    def __init__(self):
        if PSUTIL_AVAILABLE:
            self.current_process = psutil.Process(os.getpid())
        else:
            logger.warning("psutil not available, memory tracking disabled")
            self.current_process = None
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if self.current_process:
            return self.current_process.memory_info().rss / 1024 / 1024
        return 0.0
    
    @contextmanager
    def measure_agent_performance(self, agent_name: str, state: Dict[str, Any]):
        """
        Context manager to measure agent performance for LangSmith
        Enhanced version with retry context logging
        """
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Extract retry context for enhanced logging
        retry_count = state.get('retry_count', 0)
        is_retry = state.get('is_retry', False)
        retry_type = state.get('retry_type', '')
        
        # Log agent start with retry context
        logger.info(
            "agent_started",
            agent=agent_name,
            trace_id=state.get('trace_id'),
            session_id=state.get('session_id'),
            query=state.get('query', '')[:100],
            retry_count=retry_count,
            is_retry=is_retry,
            retry_type=retry_type
        )
        
        try:
            yield
            
            # Calculate metrics
            end_time = time.time()
            end_memory = self.get_memory_usage()
            latency_ms = (end_time - start_time) * 1000
            memory_delta = end_memory - start_memory
            
            # Store metrics in state for LangSmith tracing
            if 'latency_ms' not in state:
                state['latency_ms'] = {}
            if 'memory_usage' not in state:
                state['memory_usage'] = {}
            
            # Use different metric names for retry attempts
            metric_name = f"{agent_name}_retry" if is_retry else agent_name
            state['latency_ms'][metric_name] = latency_ms
            state['memory_usage'][metric_name] = end_memory
            
            # Log successful completion with retry context
            logger.info(
                "agent_completed",
                agent=agent_name,
                latency_ms=latency_ms,
                memory_mb=end_memory,
                memory_delta_mb=memory_delta,
                trace_id=state.get('trace_id'),
                session_id=state.get('session_id'),
                retry_count=retry_count,
                is_retry=is_retry,
                retry_type=retry_type
            )
            
        except Exception as e:
            # Log error with retry context
            logger.error(
                "agent_error",
                agent=agent_name,
                error=str(e),
                error_type=type(e).__name__,
                trace_id=state.get('trace_id'),
                session_id=state.get('session_id'),
                retry_count=retry_count,
                is_retry=is_retry,
                retry_type=retry_type
            )
            raise
    
    # --- SIMPLIFIED: Only LangSmith helpers and performance measurement ---
    # Individual logging methods removed to avoid duplication with agent loggers
    

# Global observability manager instance
observability = ObservabilityManager()


# LangSmith integration helpers
def add_langsmith_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add LangSmith specific context to state for better tracing
    """
    langsmith_context = {
        'langsmith_trace_id': state.get('trace_id'),
        'langsmith_session_id': state.get('session_id'),
        'langsmith_run_type': 'multi_agent_rag',
        'langsmith_tags': ['rag', 'multi-agent', 'workflow'],
        'langsmith_metadata': {
            'route': state.get('route'),
            'retry_count': state.get('retry_count', 0),
            'is_retry': state.get('is_retry', False),
            'validation_passed': state.get('validation_passed'),
            'confidence_score': state.get('confidence_score')
        }
    }
    
    # Add performance metrics if available
    if 'latency_ms' in state:
        langsmith_context['langsmith_metadata']['performance'] = {
            'total_latency_ms': sum(state['latency_ms'].values()),
            'agent_latencies': state['latency_ms']
        }
    
    if 'memory_usage' in state:
        langsmith_context['langsmith_metadata']['memory'] = state['memory_usage']
    
    return langsmith_context


def log_for_langsmith(event_name: str, state: Dict[str, Any], **kwargs):
    """
    Helper function to log events in a format optimized for LangSmith
    """
    log_data = {
        'event': event_name,
        'trace_id': state.get('trace_id'),
        'session_id': state.get('session_id'),
        'timestamp': time.time(),
        **kwargs
    }
    
    # Add LangSmith context
    langsmith_context = add_langsmith_context(state)
    log_data.update(langsmith_context)
    
    logger.info("langsmith_event", **log_data)
