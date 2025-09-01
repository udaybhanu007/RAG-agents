"""
Monkey patches for graphiti-core 0.8.1 compatibility issues.

This module contains targeted fixes for known issues in graphiti-core 0.8.1
to maintain compatibility with Azure OpenAI while using the stable version.
"""

import logging
from typing import List, Dict, Any, Optional
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Store the original method
_original_generate_response = OpenAIClient._generate_response

async def _patched_generate_response(
    self,
    messages: List[Message],
    response_model: Optional[type[BaseModel]] = None,
    max_tokens: int = 4000,  # Increased for optimal performance
) -> Dict[str, Any]:
    """
    Patched version of _generate_response that handles response_format=None properly.
    
    This fixes the "Unsupported response_format type - None" error in graphiti-core 0.8.1
    by using the regular chat/completions endpoint when no response model is provided.
    """
    from openai.types.chat import ChatCompletionMessageParam
    
    # Convert messages to OpenAI format
    openai_messages: List[ChatCompletionMessageParam] = []
    for m in messages:
        m.content = self._clean_input(m.content)
        if m.role == 'user':
            openai_messages.append({'role': 'user', 'content': m.content})
        elif m.role == 'system':
            openai_messages.append({'role': 'system', 'content': m.content})
    
    try:
        # If no response model is provided, use regular chat completions
        if response_model is None:
            # Use requested token limit with fallback to conservative value
            actual_max_tokens = max_tokens if max_tokens else 4000
            logger.info(f"🔧 Token limits - requested: {max_tokens}, self.max_tokens: {getattr(self, 'max_tokens', 'NOT_SET')}, using: {actual_max_tokens}")
            
            response = await self.client.chat.completions.create(
                model=self.model or "gpt-4",
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=actual_max_tokens,
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise Exception("No content in response")
                
            # Return in the expected format
            return {"content": content}
        else:
            # Use the original method for structured responses
            return await _original_generate_response(self, messages, response_model, max_tokens)
            
    except Exception as e:
        logger.error(f"Error in patched generate_response: {e}")
        raise


def apply_graphiti_patches():
    """Apply all necessary patches for graphiti-core 0.8.1 compatibility."""
    logger.info("Applying graphiti-core 0.8.1 compatibility patches...")
    
    # Patch the response_format issue
    OpenAIClient._generate_response = _patched_generate_response
    
    logger.info("✅ Graphiti-core patches applied successfully")
