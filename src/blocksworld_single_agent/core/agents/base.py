"""Agent Base Interface - Abstract base class for all agent implementations.

This module provides the abstract interface that all agents must implement for use
in agent networks. It defines the contract for agent initialization, tool configuration,
message handling, and resource management.

Key Design Principles:
- Abstract Interface: Defines contract for all agent implementations  
- Flexible Tool Configuration: Configurable MCP tool loading with filtering
- Agent Type Agnostic: Supports different agent architectures (ReAct, Chain, Graph)
- Network Ready: Designed for use in agent networks and multi-agent scenarios
- Resource Management: Proper cleanup and resource tracking

Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Agent Network   │ -> │ AgentInterface   │ -> │ Concrete Agent  │
│ / GUI           │    │ (abstract)       │    │ (ReAct, etc.)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                v
                       ┌─────────────────┐
                       │ MCP Tools       │
                       │ (configurable)  │
                       └─────────────────┘

Usage Example:
    class MyAgent(AgentInterface):
        async def _create_agent(self, tools):
            return create_react_agent(self.llm, tools)
    
    agent = MyAgent("gpt-4o-mini", temperature=0.1)
    await agent.initialize()
    
    async for chunk in agent.send_message("Hello"):
        print(chunk, end="")
    
    await agent.close()
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List, Optional, Set
from dataclasses import dataclass
import signal
import uuid
from langchain_core.messages import ToolMessage

# Internal dependencies
from ..models import AVAILABLE_MODELS, create_model, TokenTrackingCallback
from ..mcp import MCPManager, MCPServerConfig
from ..tool_events import ToolStartEvent, ToolEndEvent, ToolErrorEvent


@dataclass
class AgentConfig:
    """Configuration for agent initialization.
    
    This dataclass provides a flexible way to configure agents with both
    common parameters and agent-specific customizations.
    
    Attributes:
        model_name (str): Name of the LLM model to use
        temperature (float): Model temperature (0.0-1.0)
        system_prompt (str): System prompt for the agent (optional, agent-specific default if None)
        mcp_servers (List[MCPServerConfig]): MCP server configurations (optional, overrides .env)
        allowed_tool_tags (Set[str]): Set of tool tags that this agent is allowed to use.
            If None, all tools from all configured servers are available. If specified,
            only tools with matching tags will be loaded. This allows fine-grained control
            over which tools each agent can access.
            
            Examples:
                # Agent only with filesystem tools
                allowed_tool_tags = {"filesystem", "files"}
                
                # Agent only with simulation tools  
                allowed_tool_tags = {"simulation", "blocksworld"}
                
                # Agent with web and search tools
                allowed_tool_tags = {"web", "search", "internet"}
                
        agent_specific_config (Dict[str, Any]): Flexible container for agent-specific parameters.
            This dictionary allows passing custom configuration parameters to specific agent
            implementations without modifying the base interface. Each agent type can define
            its own supported parameters.
            
            Examples:
                # ReAct Agent specific parameters
                agent_specific_config = {
                    "max_iterations": 10,
                    "early_stopping_method": "generate",
                    "memory_type": "postgres",
                    "thread_id_prefix": "react_"
                }
                
                # Chain-of-Thought Agent parameters  
                agent_specific_config = {
                    "chain_length": 5,
                    "reasoning_depth": "deep",
                    "intermediate_steps": True,
                    "self_critique": True
                }
                
                # Graph-based Agent parameters
                agent_specific_config = {
                    "graph_structure": "hierarchical",
                    "node_types": ["planner", "executor", "critic"],
                    "execution_mode": "parallel",
                    "max_depth": 3
                }
                
            This design follows the Open/Closed Principle - the interface is open for
            extension (new agent-specific parameters) but closed for modification.
    """
    model_name: str
    temperature: float = 0.1
    system_prompt: Optional[str] = None
    mcp_servers: Optional[List[MCPServerConfig]] = None
    allowed_tool_tags: Optional[Set[str]] = None
    agent_specific_config: Optional[Dict[str, Any]] = None


class AgentInterface(ABC):
    """Abstract base class for all agent implementations.
    
    This class defines the contract that all agents must implement to work
    in agent networks and with the GUI. Concrete implementations handle
    specific agent types (ReAct, Chain-of-Thought, Graph-based, etc.).
    
    The interface handles:
    - Model initialization and configuration
    - MCP tool loading and filtering via shared MCPManager
    - Token usage tracking and cost calculation
    - Abstract agent creation (implemented by subclasses)  
    - Message handling with streaming support
    - Proper resource cleanup
    
    Attributes:
        model_name (str): Name of the LLM model to use
        temperature (float): Model temperature (0.0-1.0)
        token_callback (TokenTrackingCallback): Tracks usage and costs
        llm: The underlying language model instance
        config (AgentConfig): Complete agent configuration
        
    Private Attributes:
        _agent: Agent instance (type depends on implementation)
        _mcp_manager: Shared MCP manager instance
        _initialized (bool): Whether initialization completed successfully
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1, 
                 system_prompt: Optional[str] = None, config: Optional[AgentConfig] = None,
                 response_format=None):
        """Initialize the agent interface.
        
        Args:
            model_name (str): Name of the LLM model (must be in AVAILABLE_MODELS)
            temperature (float): Model temperature between 0.0 and 1.0
            system_prompt (str): Optional system prompt for the agent
            config (AgentConfig): Optional complete configuration override
            response_format: Optional response format for structured output (e.g., TypedDict)
                
        Raises:
            ValueError: If model_name is not supported or temperature is invalid
        """
        if model_name not in AVAILABLE_MODELS:
            available = ", ".join(AVAILABLE_MODELS.keys())
            raise ValueError(f"Model {model_name} not supported. Available: {available}")
            
        if not 0.0 <= temperature <= 1.0:
            raise ValueError(f"Temperature must be between 0.0 and 1.0, got {temperature}")
        
        # Use provided config or create default
        if config:
            self.config = config
        else:
            self.config = AgentConfig(
                model_name=model_name,
                temperature=temperature,
                system_prompt=system_prompt
            )
        
        # Public configuration (for backwards compatibility)
        self.model_name = self.config.model_name
        self.temperature = self.config.temperature
        self.token_callback = None
        self.llm = None
        
        # Private agent components
        self._agent = None
        self._mcp_manager = None
        self._initialized = False
        self._response_format = response_format
        
        # Interrupt handling
        self._interrupted = False
    
    def is_initialized(self) -> bool:
        """Check if the agent has been successfully initialized."""
        return self._initialized
    
    def _setup_interrupt_handler(self):
        """Setup interrupt handler for CLI usage."""
        def interrupt_handler(signum, frame):
            print("\n🛑 Interrupt received - stopping at next checkpoint...")
            self._interrupted = True
        
        # Setup signal handler for SIGINT (Ctrl+C)
        try:
            signal.signal(signal.SIGINT, interrupt_handler)
        except ValueError:
            # Signal handlers can only be set in main thread
            pass
    
    def request_interrupt(self):
        """Request interruption (for programmatic control)."""
        self._interrupted = True
    
    async def initialize(self) -> bool:
        """Initialize the agent and all its dependencies.
        
        This method handles the initialization sequence:
        1. Create the LLM model instance
        2. Get shared MCP manager instance
        3. Load and filter tools from MCP servers
        4. Create agent instance (implemented by subclasses)
        5. Set up token usage tracking
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Step 1: Create the LLM model instance
            self.llm = create_model(self.config.model_name, self.config.temperature)
            
            # Step 2: Get shared MCP manager instance
            self._mcp_manager = await MCPManager.get_instance()
            
            # Step 3: Load tools and prompts from MCP servers (shared, cached)
            all_tools = await self._get_tools_from_mcp()
            all_prompts = await self._get_prompts_from_mcp()
            
            # Step 4: Create agent instance (implemented by subclasses)
            self._agent = await self._create_agent(all_tools, all_prompts)
            
            # Step 5: Set up token usage tracking
            self.token_callback = TokenTrackingCallback(self.config.model_name)
            
            self._initialized = True
            return True
            
        except Exception as e:
            self._initialized = False
            raise Exception(f"Failed to initialize agent: {str(e)}")
    
    async def _get_tools_from_mcp(self) -> List[Any]:
        """Get tools from MCP manager (shared, cached).
        
        Returns:
            List of tools filtered by allowed_tool_tags if specified
        """
        if self.config.mcp_servers:
            # Use configured servers with tag filtering
            return await self._mcp_manager.get_tools(
                self.config.mcp_servers, 
                tool_tags=self.config.allowed_tool_tags
            )
        else:
            # Use all servers from .env with tag filtering
            return await self._mcp_manager.get_all_tools(
                allowed_tags=self.config.allowed_tool_tags
            )
    
    async def _get_prompts_from_mcp(self) -> List[Any]:
        """Get prompts from MCP manager (shared, cached) with tag filtering.
        
        Returns:
            List of prompts filtered by allowed_tool_tags if specified
        """
        if self.config.mcp_servers:
            # Use configured servers with tag filtering
            return await self._mcp_manager.get_prompts(
                self.config.mcp_servers, 
                prompt_tags=self.config.allowed_tool_tags
            )
        else:
            # Use all servers from .env with tag filtering
            return await self._mcp_manager.get_all_prompts(
                allowed_tags=self.config.allowed_tool_tags
            )
    
    @abstractmethod
    async def _create_agent(self, tools: List[Any], prompts: List[Any] = None) -> Any:
        """Create the agent instance. Must be implemented by subclasses.
        
        Args:
            tools (List[Any]): List of available tools from MCP servers
            prompts (List[Any]): List of available prompts from MCP servers (optional)
            
        Returns:
            Any: Agent instance suitable for the specific implementation
        """
        pass
    
    async def send_message(self, message: str, thread_id: str = "") -> AsyncGenerator[str, None]:
        """Send a message to the agent and stream back the response with interrupt support.
        
        This default implementation works with LangGraph agents that support
        astream_events. Override in subclasses for different streaming approaches.
        
        Args:
            message (str): User message to send to the agent
            thread_id (str, optional): Custom thread ID for memory isolation. 
                                     If None, uses default thread ID "1"
            
        Yields:
            str: Response chunks from the agent as they arrive
            
        Raises:
            RuntimeError: If agent is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call await agent.initialize() first.")
        
        # Setup interrupt handling
        self._interrupted = False
        self._setup_interrupt_handler()
        
        try:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 100}
            async for event in self._agent.astream_events(
                {"messages": [{"role": "user", "content": message}]},
                version="v2", 
                config=config
            ):
                # Check for interrupt before processing each event
                if self._interrupted:
                    yield "\n🛑 Execution interrupted by user\n"
                    return
                    
                async for chunk in self._process_streaming_event(event):
                    yield chunk
                                        
        except KeyboardInterrupt:
            yield "\n🛑 Execution interrupted by Ctrl+C\n"
        except Exception as e:
            yield f"Error: {str(e)}"
        finally:
            # Reset interrupt flag
            self._interrupted = False
    
    async def send_message_with_structured_response(self, message: str, thread_id: str = None):
        """Send a message and return both streaming text and structured response with interrupt support.
        
        This method provides both real-time streaming (for user feedback) and access
        to structured response data (when response_format is configured).
        
        Args:
            message (str): User message to send to the agent
            thread_id (str, optional): Custom thread ID for memory isolation. 
                                     If None, uses default thread ID "1"
            
        Returns:
            Tuple of (text_chunks_list, structured_response_dict)
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call await agent.initialize() first.")
        
        # Setup interrupt handling
        self._interrupted = False
        self._setup_interrupt_handler()
        
        try:
            config = {"configurable": {"thread_id": thread_id or "1"}, "recursion_limit": 200}
            text_chunks = []
            structured_response = None
            
            # Use the same streaming approach as send_message for consistency
            async for event in self._agent.astream_events(
                {"messages": [{"role": "user", "content": message}]},
                version="v2", 
                config=config
            ):
                # Check for interrupt before processing each event
                if self._interrupted:
                    text_chunks.append("\n🛑 Execution interrupted by user\n")
                    break
                    
                # Extract structured_response from events if available
                if event.get("event") == "on_chain_end":
                    data = event.get("data", {})
                    output = data.get("output", {})
                    if isinstance(output, dict) and 'structured_response' in output:
                        structured_response = output['structured_response']
                
                # Process streaming events and collect chunks
                async for chunk in self._process_streaming_event(event):
                    if chunk:  # Only add non-empty chunks
                        text_chunks.append(chunk)
                        
            return text_chunks, structured_response
            
        except KeyboardInterrupt:
            text_chunks.append("\n🛑 Execution interrupted by Ctrl+C\n")
            return text_chunks, structured_response
        except Exception as e:
            return [f"Error: {str(e)}"], None
        finally:
            # Reset interrupt flag
            self._interrupted = False
    
    
    async def _process_streaming_event(self, event: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Process a streaming event and yield response chunks.
        
        Args:
            event (Dict[str, Any]): Event data from LangGraph streaming
            
        Yields:
            str: Response chunks to display to user
            
        Override in subclasses for custom event processing.
        """
        event_type = event.get("event")
        
        if event_type == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                if isinstance(content, str):
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            yield block.get('text', '')
                        elif hasattr(block, 'text'):
                            yield block.text
                else:
                    yield str(content)
        
        elif event_type == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            if output:
                # Handle token tracking
                if hasattr(output, 'usage_metadata') and output.usage_metadata and self.token_callback:
                    usage = output.usage_metadata
                    self.token_callback.update_from_usage(
                        prompt_tokens=usage.get('input_tokens', 0),
                        completion_tokens=usage.get('output_tokens', 0)
                    )
                
                # For non-streaming models (o3), output the content here
                if self.model_name == "o3":
                    # This is a non-streaming model, output content here
                    if hasattr(output, 'content'):
                        content = output.content
                        # Handle structured content (list of blocks)
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and 'text' in block:
                                    yield block['text']
                                elif hasattr(block, 'text'):
                                    yield block.text
                        else:
                            yield str(content)
                    elif isinstance(output, str):
                        yield output
        
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "tool")
            tool_input = event.get("data", {}).get("input", {})
            # Send structured tool start event
            event_obj = ToolStartEvent(tool_name=tool_name, tool_input=tool_input)
            yield event_obj.to_string()
        
        elif event_type == "on_tool_end":
            tool_name = event.get("name", "tool")
            output = event.get("data", {}).get("output", "")
            # Extract only content from tool output
            if hasattr(output, 'content'):
                tool_content = output.content
            else:
                tool_content = str(output)
            # Send structured tool end event
            event_obj = ToolEndEvent(tool_name=tool_name, tool_output=tool_content)
            yield event_obj.to_string()
        
        elif event_type == "on_chain_stream":
            # Check for ToolMessage with error status in chain stream
            data = event.get("data", {})
            chunk = data.get("chunk", {})
            
            # Look for messages in the chunk
            if "messages" in chunk:
                for message in chunk["messages"]:
                    # Check if this is a ToolMessage with error status
                    is_tool_message = (
                        (ToolMessage and isinstance(message, ToolMessage)) or
                        (hasattr(message, '__class__') and message.__class__.__name__ == 'ToolMessage')
                    )
                    
                    if (is_tool_message and 
                        hasattr(message, 'status') and message.status == 'error'):
                        
                        tool_name = getattr(message, 'name', 'unknown_tool')
                        error_content = getattr(message, 'content', '')
                        
                        # Send structured tool error event
                        event_obj = ToolErrorEvent(tool_name=tool_name, error_message=error_content)
                        yield event_obj.to_string()
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics for the agent session.
        
        Note: In multi-agent scenarios, this returns stats for this specific agent.
        Use a higher-level tracker for network-wide statistics.
        """
        if self.token_callback:
            return {
                "total_tokens": self.token_callback.total_tokens,
                "input_tokens": self.token_callback.prompt_tokens,
                "output_tokens": self.token_callback.completion_tokens,
                "total_cost": self.token_callback.total_cost,
                "model_name": self.config.model_name,
                "agent_type": self.__class__.__name__
            }
        return {
            "total_tokens": 0, 
            "total_cost": 0.0, 
            "model_name": self.config.model_name,
            "agent_type": self.__class__.__name__
        }
    
    async def close(self):
        """Close the agent and cleanup resources.
        
        Note: MCP connections are shared across agents and managed by MCPManager.
        Individual agents don't close MCP connections directly.
        """
        self._initialized = False
        self._agent = None
        self._mcp_manager = None