"""
LangSmith observability configuration for multi-agent RAG workflows
"""

import os
from typing import Dict, Any
from langsmith import Client
from langsmith.run_helpers import traceable
import sys
import os
# Add the parent directory to sys.path to import from agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.logging_config import get_logger
from .azure_keyvault_manager import get_secret_from_keyvault
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
        api_key=get_secret_from_keyvault("LANGCHAIN_API_KEY"),       
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


# Export traceable for easy import
__all__ = ["traceable", "get_traceable_config", "LANGSMITH_CONFIG"]
