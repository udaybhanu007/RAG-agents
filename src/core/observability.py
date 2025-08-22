"""
LangSmith observability configuration for multi-agent RAG workflows
"""

import os
import ssl
import urllib3
from typing import Dict, Any
from langsmith import Client
from langsmith.run_helpers import traceable
import sys
import os

# Disable SSL verification for LangSmith connections
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_VERIFY'] = 'false'
os.environ['PYTHONHTTPSVERIFY'] = '0'

# Simple logging setup to avoid circular imports
import logging
logger = logging.getLogger("observability")

from .azure_keyvault_manager import get_secret_from_keyvault

# LangSmith Configuration
LANGSMITH_CONFIG = {
    "project_name": os.environ.get("LANGCHAIN_PROJECT", "hybrid-agent-rag-workflow"),
    "hide_inputs": True,      # Hide medical data for privacy
    "hide_outputs": True,     # Hide medical responses for privacy
    "auto_batch_tracing": True,
    "sample_rate": 0.5,      # 50% sampling for POC
}

# Initialize LangSmith client with SSL bypass
try:
    # Apply SSL bypass for LangSmith client
    import requests
    
    # Patch requests session to disable SSL verification for LangSmith
    original_request = requests.Session.request
    def patched_request(self, method, url, **kwargs):
        kwargs.setdefault('verify', False)
        return original_request(self, method, url, **kwargs)
    requests.Session.request = patched_request # type: ignore
    
    langsmith_api_key = get_secret_from_keyvault("LANGCHAIN_API_KEY")
    langsmith_endpoint = get_secret_from_keyvault("LANGCHAIN_ENDPOINT")
    
    # Set environment variables for LangSmith SDK
    if langsmith_api_key and langsmith_endpoint:
        os.environ['LANGCHAIN_API_KEY'] = langsmith_api_key
        os.environ['LANGCHAIN_ENDPOINT'] = langsmith_endpoint
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        
        langsmith_client = Client(
            api_key=langsmith_api_key,
            api_url=langsmith_endpoint
        )
        logger.info("LangSmith client initialized successfully with SSL bypass")
    else:
        raise ValueError("LANGCHAIN_API_KEY and LANGCHAIN_ENDPOINT must be available")
except Exception as e:
    logger.error(f"Failed to initialize LangSmith client: {e}")
    logger.info("Continuing without LangSmith tracing...")
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
            "project": os.environ.get("LANGCHAIN_PROJECT", "hybrid-agent-rag-workflow")
        }
    }


# Export for easy import
__all__ = ["traceable", "get_traceable_config", "LANGSMITH_CONFIG"]
