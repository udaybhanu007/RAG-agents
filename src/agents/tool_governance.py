from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import uuid
from datetime import datetime
from .workflow_state import WorkflowState

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

class StateValidationError(SecurityError):
    pass

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.metadata: Dict[str, ToolMetadata] = {}
        self.logger = logging.getLogger("tool_governance")
        self.agent_sessions: Dict[str, Dict] = {}  # Track active agent sessions
    
    def register_tool(self, tool_func: Callable, metadata: ToolMetadata):
        """Register a tool with access control metadata"""
        self.tools[metadata.name] = tool_func
        self.metadata[metadata.name] = metadata
        self.logger.info(f"Tool registered: {metadata.name}")
    
    def verify_agent_permissions(self, tool_name: str, agent_role: AgentRole, agent_id: str) -> bool:
        """Verify agent has permission for tool before invocation"""
        metadata = self.metadata.get(tool_name)
        
        if not metadata:
            self.logger.error(f"Tool {tool_name} not registered")
            return False
        
        # Check role-based permissions
        if agent_role not in metadata.allowed_roles:
            self.logger.warning(f"Permission denied: {agent_role.value} (ID: {agent_id}) attempted to use {tool_name}")
            return False
        
        return True
    
    def invoke_tool(self, tool_name: str, agent_role: AgentRole, agent_id: str, params: dict) -> Any:
        """Invoke tool with pre-validated permissions"""
        # Pre-check permissions
        if not self.verify_agent_permissions(tool_name, agent_role, agent_id):
            raise AccessDeniedError(f"Agent {agent_role.value} (ID: {agent_id}) not authorized for tool {tool_name}")
        
        # Execute tool
        try:
            self.logger.info(f"Executing tool: {tool_name} by {agent_role.value} (ID: {agent_id})")
            return self.tools[tool_name].invoke(params) # type: ignore
        except Exception as e:
            self.logger.error(f"Tool execution failed: {tool_name} by {agent_id} - {str(e)}")
            raise
    
    def register_agent_session(self, agent_id: str, agent_role: AgentRole):
        """Register an active agent session"""
        self.agent_sessions[agent_id] = {
            'role': agent_role,
            'created_at': datetime.utcnow().isoformat(),
            'tool_access_count': 0
        }
    
    def is_agent_session_valid(self, agent_id: str) -> bool:
        """Check if agent session is valid"""
        return agent_id in self.agent_sessions

# Global tool registry instance
tool_registry = ToolRegistry()


class SimpleStateManager:
    """Simple state management without complex integrity checks"""
    
    def __init__(self):
        self.allowed_transitions = {
            "orchestrator": ["vector_rag", "graph_rag", "none"],
            "vector_rag": ["validator", "graph_rag"],
            "graph_rag": ["validator"],
            "validator": ["synthesizer"],
            "synthesizer": ["end"]
        }
        self.logger = logging.getLogger("state_manager")
    
    def validate_state_transition(self, current_agent: str, next_agent: str, trace_id: str) -> bool:
        """Simple state transition validation"""
        allowed_next = self.allowed_transitions.get(current_agent, [])
        is_valid = next_agent in allowed_next
        
        if not is_valid:
            self.logger.warning(
                f"Invalid state transition: {current_agent} -> {next_agent} (trace: {trace_id})"
            )
        else:
            self.logger.info(
                f"Valid state transition: {current_agent} -> {next_agent} (trace: {trace_id})"
            )
        
        return is_valid
    
    def add_simple_metadata(self, state: WorkflowState, agent_id: str) -> WorkflowState:
        """Add simple tracking metadata to state"""
        existing_metadata = state.get('_metadata') or {}
        state['_metadata'] = {
            'last_modified_by': agent_id,
            'last_modified_at': datetime.utcnow().isoformat(),
            'modification_count': existing_metadata.get('modification_count', 0) + 1
        }
        return state

# Global state manager instance
state_manager = SimpleStateManager()


class SecureAgentBase:
    """Simplified secure base class for agents"""
    
    def __init__(self, role: AgentRole):
        self.role = role
        self.agent_id = f"{role.value}_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(f"agent_{role.value}")
        
        # Register agent session
        tool_registry.register_agent_session(self.agent_id, self.role)
        
        self.logger.info(f"Agent initialized: {self.agent_id} with role {role.value}")
    
    def check_tool_permission(self, tool_name: str) -> bool:
        """Check if agent has permission for tool before using it"""
        return tool_registry.verify_agent_permissions(tool_name, self.role, self.agent_id)
    
    def invoke_tool(self, tool_name: str, params: dict):
        """Secure tool invocation with pre-permission check"""
        # Validate session first
        if not tool_registry.is_agent_session_valid(self.agent_id):
            raise AccessDeniedError(f"Invalid agent session: {self.agent_id}")
        
        # Pre-check permissions before invocation
        if not self.check_tool_permission(tool_name):
            raise AccessDeniedError(f"Agent {self.role.value} (ID: {self.agent_id}) lacks permission for tool {tool_name}")
        
        try:
            # Increment tool access count
            tool_registry.agent_sessions[self.agent_id]['tool_access_count'] += 1
            
            # Invoke tool with agent identification
            return tool_registry.invoke_tool(tool_name, self.role, self.agent_id, params)
            
        except AccessDeniedError as e:
            self.logger.error(f"Tool access denied for {self.role.value}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Tool execution error for {self.role.value}: {str(e)}")
            raise
    
    def update_state_and_transition(self, state: WorkflowState, next_agent: str) -> WorkflowState:
        """Update state with simple metadata and validate transition"""
        trace_id = state.get('trace_id') or 'unknown'
        
        # Simple transition validation
        if not state_manager.validate_state_transition(self.role.value, next_agent, trace_id):
            raise StateValidationError(f"Invalid state transition from {self.role.value} to {next_agent}")
        
        # Add simple metadata
        updated_state = state_manager.add_simple_metadata(state, self.agent_id)
        
        self.logger.info(f"State updated by {self.agent_id}, transitioning to {next_agent}")
        return updated_state
