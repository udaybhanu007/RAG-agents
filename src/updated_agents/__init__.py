"""
Simple Agentic RAG System

This package contains the simplified agentic-enhanced RAG system with:
- TRUE autonomous behavior with reasoning and learning
- Simple, modular architecture following the roadmap requirements
- Self-contained implementation without external agent dependencies
- Maximum code reuse with minimal complexity

Key Components:
- base_classes: Foundational classes and utilities
- simple_agentic_agents: Core agentic agents with reasoning capabilities
- simple_agentic_app: Main application interface
- simple_agentic_streamlit: Streamlit frontend for user interaction
"""

import os
import sys

# Add the src directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from core.logging_config import get_logger

# Initialize package logger
logger = get_logger("updated_agents")
logger.info("simple_agentic_package_imported")

from .base_classes import (
    WorkflowState,
    SecureAgentBase,
    AgentRole,
    QueryAnalysis,
    RoutingDecision,
    ValidationResult,
    SynthesisResult,
    tool_registry
)

from .simple_agentic_agents import (
    LearningMemory,
    SimpleReasoningPlan,
    AgenticOrchestratorAgent,
    AgenticVectorRAGAgent,
    AgenticGraphRAGAgent,
    SimpleValidatorAgent,
    SimpleAnswerSynthesisAgent,
    SimpleAgenticWorkflow,
    create_simple_agentic_workflow
)

from .simple_agentic_app import (
    SimpleAgenticRAGApplication,
    simple_agentic_app
)

__all__ = [
    # Base classes
    'WorkflowState',
    'SecureAgentBase', 
    'AgentRole',
    'QueryAnalysis',
    'RoutingDecision',
    'ValidationResult',
    'SynthesisResult',
    'tool_registry',
    
    # Simple agentic agents
    'LearningMemory',
    'SimpleReasoningPlan',
    'AgenticOrchestratorAgent',
    'AgenticVectorRAGAgent', 
    'AgenticGraphRAGAgent',
    'SimpleValidatorAgent',
    'SimpleAnswerSynthesisAgent',
    'SimpleAgenticWorkflow',
    'create_simple_agentic_workflow',
    
    # Main application
    'SimpleAgenticRAGApplication',
    'simple_agentic_app'
]
