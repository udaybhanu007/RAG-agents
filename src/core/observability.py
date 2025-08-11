"""
LangSmith observability configuration for multi-agent RAG workflows
"""

import os
from typing import Dict, Any
from langsmith import Client
from langsmith.run_helpers import traceable
from logging_config import get_logger

logger = get_logger("observability")

# LangSmith Configuration
LANGSMITH_CONFIG = {
    "project_name": "Agentic-RAG-Workflow",
    "hide_inputs": True,      # Hide medical data for privacy
    "hide_outputs": True,     # Hide medical responses for privacy
    "auto_batch_tracing": True,
    "sample_rate": 0.5,      # 50% sampling for POC
}

# Initialize LangSmith client
try:
    langsmith_client = Client(
        api_key=os.getenv("LANGCHAIN_API_KEY"),
        **LANGSMITH_CONFIG
    )
    logger.info("LangSmith client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LangSmith client: {e}")
    langsmith_client = None


def get_traceable_config(agent_name: str) -> Dict[str, Any]:
    """
    Get LangSmith traceable configuration for specific agent
    """
    return {
        "name": agent_name,
        "hide_inputs": True,
        "hide_outputs": True,
        "metadata": {
            "agent_type": agent_name,
            "project": "Agentic-RAG-Workflow"
        }
    }


class ObservabilityManager:
    """Manager for observability operations - simplified for LangSmith-only monitoring"""
    pass


# Create global observability instance
observability = ObservabilityManager()


# Export traceable for easy import
__all__ = ["traceable", "get_traceable_config", "LANGSMITH_CONFIG", "observability"]
