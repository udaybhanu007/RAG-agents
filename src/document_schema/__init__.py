"""
Document Schema Module for Graphiti Ingestion

Provides simple document container for Graphiti's dynamic schema evolution.
No manual entity extraction or schema patterns - Graphiti handles this automatically.
"""

from .base_schema import DocumentContainer

__all__ = ["DocumentContainer"]

__version__ = "1.0.0"
__author__ = "RAG Agents Team"
