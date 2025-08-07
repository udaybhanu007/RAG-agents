from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging

class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    VECTOR_RAG = "vector_rag"
    GRAPH_RAG = "graph_rag"
    VALIDATOR = "validator"
    SYNTHESIZER = "synthesizer"

@dataclass
class ToolMetadata:
    name: str
    allowed_roles: List[AgentRole]

class SecurityError(Exception):
    pass

class AccessDeniedError(SecurityError):
    pass

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.logger = logging.getLogger("tool_governance")
    
    def register_tool(self, tool_func: Callable, metadata: ToolMetadata):
        """Register a tool with access control metadata"""
        self.tools[metadata.name] = tool_func
        self.metadata[metadata.name] = metadata
        self.logger.info(f"Tool registered: {metadata.name}")
    
    def invoke_tool(self, tool_name: str, agent_role: AgentRole, params: dict) -> Any:
        """Invoke tool with access control check"""
        metadata = self.metadata.get(tool_name)
        
        if not metadata:
            raise SecurityError(f"Tool {tool_name} not registered")
        
        # Access control check
        if agent_role not in metadata.allowed_roles:
            self.logger.warning(f"Access denied: {agent_role.value} attempted to use {tool_name}")
            raise AccessDeniedError(f"Agent {agent_role.value} not authorized for tool {tool_name}")
        
        # Execute tool
        try:
            return self.tools[tool_name].invoke(params)
        except Exception as e:
            self.logger.error(f"Tool execution failed: {tool_name} - {str(e)}")
            raise

# Global tool registry instance
tool_registry = ToolRegistry()


class SecureAgentBase:
    """Base class for agents with tool governance"""
    def __init__(self, role: AgentRole):
        self.role = role
    
    def invoke_tool(self, tool_name: str, params: dict):
        """Secure tool invocation through registry"""
        try:
            return tool_registry.invoke_tool(tool_name, self.role, params)
        except AccessDeniedError as e:
            import logging
            logger = logging.getLogger("tool_governance")
            logger.error(f"Tool access denied for {self.role.value}: {str(e)}")
            raise
