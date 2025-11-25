"""Simple LLM Agent - Direct LLM interface without ReAct framework.

This module implements a simplified agent that provides direct access to the LLM
without the ReAct framework overhead. It's ideal for simple tasks like classification,
routing, or direct Q&A where tools are not needed.

Key Features:
- Direct LLM interaction without ReAct framework
- No tool integration (faster for simple tasks)
- Streaming response support
- Token tracking and cost calculation
- System prompt configuration
- Structured output support

Usage:
    agent = SimpleLLMAgent("gpt-4o-mini", temperature=0.1)
    await agent.initialize()
    
    async for chunk in agent.send_message("Classify this task: Move block A"):
        print(chunk, end="")
    
    await agent.close()
"""

from typing import AsyncGenerator, Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from .base import AgentInterface


class SimpleLLMAgent(AgentInterface):
    """Simple LLM Agent without ReAct framework.
    
    This agent provides direct access to the underlying LLM without the ReAct
    framework overhead. It's optimized for simple tasks like classification,
    routing, or direct question-answering where tools are not needed.
    
    Benefits:
    - Faster execution (no ReAct reasoning loop)
    - Lower token usage (no tool descriptions)
    - Direct LLM response streaming
    - Simpler for classification tasks
    
    Use Cases:
    - Task classification and routing
    - Simple Q&A without tools
    - Text classification and analysis
    - Direct LLM prompting scenarios
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1, 
                 system_prompt: Optional[str] = None, response_format=None):
        """Initialize the simple LLM agent.
        
        Args:
            model_name (str): Name of the LLM model to use
            temperature (float): Model temperature between 0.0 and 1.0
            system_prompt (str): Optional system prompt for the agent
            response_format: Optional response format for structured output
        """
        super().__init__(model_name, temperature, system_prompt, response_format=response_format)
        
        # Simple LLM agents don't need tools, so we don't configure MCP
        self.config.allowed_tool_tags = set()  # No tools needed
    
    async def _create_agent(self, tools: List[Any], prompts: List[Any] = None) -> Any:
        """Create the simple LLM agent (just returns the LLM instance).
        
        Args:
            tools (List[Any]): Not used for simple LLM agent
            prompts (List[Any]): Not used for simple LLM agent
            
        Returns:
            The LLM instance directly (no ReAct wrapper)
        """
        # For simple LLM agent, we just return the LLM directly
        # No ReAct framework, no tools, just direct LLM access
        return self.llm
    
    async def send_message(self, message: str) -> AsyncGenerator[str, None]:
        """Send a message to the LLM and stream back the response.
        
        Args:
            message (str): User message to send to the LLM
            
        Yields:
            str: Response chunks from the LLM as they arrive
            
        Raises:
            RuntimeError: If agent is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call await agent.initialize() first.")
        
        try:
            # Build messages list
            messages = []
            
            # Add system message if configured
            if self.config.system_prompt:
                messages.append(SystemMessage(content=self.config.system_prompt))
            
            # Add user message
            messages.append(HumanMessage(content=message))
            
            # Stream response directly from LLM
            async for chunk in self._agent.astream(messages):
                if chunk.content:
                    content = chunk.content
                    # Handle structured content (list of blocks) for O-series models
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and 'text' in block:
                                yield block['text']
                            elif hasattr(block, 'text'):
                                yield block.text
                    else:
                        yield str(content)
                    
                # Track token usage if available
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata and self.token_callback:
                    usage = chunk.usage_metadata
                    self.token_callback.update_from_usage(
                        prompt_tokens=usage.get('input_tokens', 0),
                        completion_tokens=usage.get('output_tokens', 0)
                    )
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def send_message_with_structured_response(self, message: str):
        """Send a message and return both streaming text and structured response.
        
        Note: For simple LLM agents, structured response is not typically used,
        but this method maintains interface compatibility.
        
        Args:
            message (str): User message to send to the agent
            
        Returns:
            Tuple of (text_chunks_list, None) - no structured response for simple LLM
        """
        text_chunks = []
        async for chunk in self.send_message(message):
            if chunk:
                text_chunks.append(chunk)
        
        # Simple LLM agents don't typically use structured responses
        return text_chunks, None
    
    async def get_single_response(self, message: str) -> str:
        """Get a complete response as a single string (non-streaming).
        
        This is a convenience method for cases where you want the full response
        at once rather than streaming chunks.
        
        Args:
            message (str): User message to send to the LLM
            
        Returns:
            str: Complete response from the LLM
        """
        chunks = []
        async for chunk in self.send_message(message):
            chunks.append(chunk)
        return "".join(chunks)
    
    async def _process_streaming_event(self, event: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Process streaming events (not used for simple LLM agent).
        
        This method is inherited from AgentInterface but not used since we
        stream directly from the LLM in send_message().
        """
        # Not used for simple LLM agent - we handle streaming directly
        return
        yield  # Make this a generator function