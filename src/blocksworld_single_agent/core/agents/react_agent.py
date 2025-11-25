"""ReAct Agent - Concrete implementation of AgentInterface using ReAct pattern.

This module provides a ReAct (Reasoning + Acting) agent implementation that uses
the LangGraph create_react_agent function. ReAct agents can reason about problems
and take actions using available tools in an iterative manner.

The ReAct pattern works by:
1. Reasoning: The agent thinks about what to do next
2. Acting: The agent uses tools to gather information or perform actions  
3. Observing: The agent observes the results and continues reasoning
4. Repeat until the problem is solved

Usage Example:
    from .react_agent import ReActAgent
    from .base import AgentConfig
    from ..mcp import MCPServerConfig
    
    # Simple usage (backwards compatible)
    agent = ReActAgent("gpt-4o-mini", temperature=0.1)
    await agent.initialize()
    
    # Usage with custom system prompt
    agent = ReActAgent("gpt-4o-mini", system_prompt="Du bist ein Experte für...")
    await agent.initialize()
    
    # Advanced usage with tool filtering
    agent = ReActAgent("gpt-4o-mini", allowed_tool_tags={"simulation", "planning"})
    await agent.initialize()
    
    # Advanced usage with custom configuration
    config = AgentConfig(
        model_name="gpt-4o-mini",
        temperature=0.1,
        system_prompt="You are a specialized planning agent...",
        allowed_tool_tags={"simulation"},
        agent_specific_config={
            "max_iterations": 15,
            "early_stopping_method": "generate"
        }
    )
    agent = ReActAgent(config=config)
    await agent.initialize()
"""

from typing import List, Optional, Set
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from .base import AgentInterface, AgentConfig


class ReActAgent(AgentInterface):
    """ReAct (Reasoning + Acting) agent implementation.
    
    This agent uses the LangGraph create_react_agent function to create an agent
    that can reason about problems and use tools to solve them iteratively.
    
    The agent maintains conversation memory using an InMemorySaver checkpointer,
    allowing it to maintain context across multiple interactions.
    
    Key Features:
    - ReAct pattern: Reasoning followed by Acting
    - Tool usage: Can use any tools provided via MCP servers
    - Memory persistence: Remembers conversation history
    - Streaming support: Real-time response streaming
    - Error handling: Graceful error recovery
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1, 
                 system_prompt: Optional[str] = None, config: Optional[AgentConfig] = None, 
                 allowed_tool_tags: Optional[Set[str]] = None, response_format=None):
        """Initialize the ReAct agent.
        
        Args:
            model_name (str): Name of the LLM model to use
            temperature (float): Model temperature (0.0-1.0)
            system_prompt (str): Optional system prompt (uses ReAct default if None)
            config (AgentConfig): Optional complete configuration
            allowed_tool_tags (Set[str]): Set of tool tags to filter by (simple interface)
            response_format: Optional response format for structured output (e.g., TypedDict)
        """
        # If allowed_tool_tags is provided but no config, create config with tags
        if allowed_tool_tags and not config:
            config = AgentConfig(
                model_name=model_name,
                temperature=temperature,
                system_prompt=system_prompt,
                allowed_tool_tags=allowed_tool_tags
            )
        
        super().__init__(model_name, temperature, system_prompt, config, response_format)
        self._memory = None
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for ReAct agents.
        
        Returns:
            str: Default system prompt optimized for ReAct pattern
        """
        return (
        """
        You are a reasoning and planning AI assistant that can use tools step by step to solve tasks.

        When a user provides an input goal, follow this structured process:

        1. Use the `get_rules` tool to understand the environment rules and constraints.
        2. Use 'get_status' to check the current system state and gather information.
        3. Create a plan to reach the user’s goal. The plan must strictly follow all rules and avoid illegal actions.
        4. Verify the created plan using the `verify_plan` tool.
        - The verification tool provides feedback on which step of the plan is invalid.
        5. If the plan is invalid, correct only the failing parts based on feedback from 'verify_plan' tool and re-verify.
        6. Once the plan is verified as valid, execute it using available tools.
        7. After execution, confirm whether the user’s goal has been fully achieved.
        """
        )
    
    async def _create_agent(self, tools: List, prompts: List = []):
        """Create a ReAct agent with the provided tools.
        
        Args:
            tools: List of available tools from MCP servers
            prompts: List of available prompts (unused, available via _mcp_manager)
            
        Returns:
            LangGraph ReAct agent instance
        """
        # Initialize memory saver for agent state persistence
        self._memory = InMemorySaver()
        
        # Apply agent-specific configuration if provided
        agent_kwargs = {
            "model": self.llm,
            "tools": tools,
            "checkpointer": self._memory,
        }
        
        # Add response_format if specified
        if self._response_format is not None:
            agent_kwargs["response_format"] = self._response_format
        
        # Add ReAct-specific configurations
        if self.config.agent_specific_config:
            specific_config = self.config.agent_specific_config
            
            # Map common ReAct parameters
            if "max_iterations" in specific_config:
                agent_kwargs["max_iterations"] = specific_config["max_iterations"]
            
            if "early_stopping_method" in specific_config:
                agent_kwargs["early_stopping_method"] = specific_config["early_stopping_method"]
        
        # Create ReAct agent with configured parameters
        return create_react_agent(**agent_kwargs)
    
    def get_memory_stats(self) -> dict:
        """Get memory-related statistics for this ReAct agent.
        
        Returns:
            Dict with memory information including ReAct-specific details
        """
        stats = self.get_stats()
        stats.update({
            "has_memory": self._memory is not None,
            "memory_type": "InMemorySaver" if self._memory else None,
            "agent_pattern": "ReAct",
            "supports_tools": True,
            "supports_memory": True
        })
        return stats