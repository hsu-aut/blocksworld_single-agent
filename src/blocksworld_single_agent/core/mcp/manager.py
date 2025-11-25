"""MCP Manager - Centralized management of MCP servers and tools.

This module provides centralized management of MCP (Model Context Protocol) servers
and tools, allowing multiple agents to share the same server instances and tools
efficiently. This avoids redundant server startups and resource usage.

Key Features:
- Singleton pattern for shared MCP server instances
- Tool caching and filtering by tags
- Connection pooling and lifecycle management
- Graceful error handling and recovery
- Resource cleanup on shutdown

Architecture:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Agent 1         │ -> │ MCPManager       │ -> │ MCP Server A    │
├─────────────────┤    │ (Singleton)      │    ├─────────────────┤
│ Agent 2         │ -> │                  │ -> │ MCP Server B    │
├─────────────────┤    │ - Tool Cache     │    ├─────────────────┤
│ Agent N         │ -> │ - Connection Pool│ -> │ MCP Server N    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
"""

import os
import asyncio
from typing import Dict, List, Set, Optional, Any
from dotenv import load_dotenv

# MCP dependencies
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from .config import MCPServerConfig, MCPConnection


class MCPManager:
    """Singleton manager for MCP servers and tools.
    
    This class manages MCP server connections and provides tools to agents
    on demand. It ensures that each unique MCP server is only started once
    and tools are cached for efficient reuse.
    
    Features:
    - Singleton pattern: One instance per application
    - Connection pooling: Reuse server connections
    - Tool caching: Cache tools by server and tags  
    - Lifecycle management: Proper startup and shutdown
    - Error resilience: Continue with available servers if some fail
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MCPManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.connections: Dict[str, MCPConnection] = {}
            self.tool_cache: Dict[str, List[Any]] = {}
            self.prompt_cache: Dict[str, List[Any]] = {}
            self._lock = asyncio.Lock()
            MCPManager._initialized = True
    
    @classmethod 
    async def get_instance(cls) -> 'MCPManager':
        """Get or create the singleton MCPManager instance."""
        instance = cls()
        if not instance.connections:
            await instance._load_environment()
        return instance
    
    async def _load_environment(self):
        """Load environment variables once."""
        load_dotenv()
    
    async def initialize_servers(self, server_configs: List[MCPServerConfig]) -> bool:
        """Initialize MCP servers from configurations.
        
        Args:
            server_configs: List of server configurations
            
        Returns:
            bool: True if at least one server initialized successfully
        """
        async with self._lock:
            successful_connections = 0
            
            for config in server_configs:
                server_key = self._get_server_key(config)
                
                # Check if connection exists and is still valid
                if server_key in self.connections:
                    connection = self.connections[server_key]
                    if await self._is_connection_valid(connection):
                        successful_connections += 1
                        continue
                    else:
                        # Remove invalid connection
                        await self._close_connection_safely(server_key, connection)
                        del self.connections[server_key]
                        if server_key in self.tool_cache:
                            del self.tool_cache[server_key]
                        if server_key in self.prompt_cache:
                            del self.prompt_cache[server_key]
                
                try:
                    connection = await self._connect_to_server(config)
                    self.connections[server_key] = connection
                    successful_connections += 1
                    
                    # Cache tools and prompts for this server
                    self.tool_cache[server_key] = connection.tools
                    self.prompt_cache[server_key] = connection.prompts
                    
                except Exception as e:
                    print(f"⚠️  Failed to connect to MCP server {config.name or config.command}: {e}")
                    continue
            
            return successful_connections > 0
    
    async def _is_connection_valid(self, connection: MCPConnection) -> bool:
        """Check if an MCP connection is still valid."""
        try:
            # Simple check: see if the session is still accessible
            if not hasattr(connection, 'mcp_session') or not connection.mcp_session:
                return False
            
            # Try to list resources (this is a basic "ping" test)
            # This will fail if the connection is broken
            await connection.mcp_session.list_resources()
            return True
        except Exception:
            return False

    async def _connect_to_server(self, config: MCPServerConfig) -> MCPConnection:
        """Connect to a single MCP server."""
        server_params = StdioServerParameters(
            command=config.command,
            args=config.args
        )
        
        # Initialize MCP connection
        client_context = stdio_client(server_params)
        stdio_session = await client_context.__aenter__()
        session_context = ClientSession(*stdio_session)
        mcp_session = await session_context.__aenter__()
        await mcp_session.initialize()
        
        # Load tools from server
        tools = await load_mcp_tools(mcp_session)
        
        # NOTE: MCP Protocol doesn't support tag filtering yet (GitHub Issue #522)
        # FastMCP supports @mcp.tool(tags=...) but list_tools() doesn't filter by tags
        # For now, all tools are loaded regardless of tags
        
        # Load prompts from server
        prompts = []
        try:
            # Get available prompts from server
            prompts_result = await mcp_session.list_prompts()
            if hasattr(prompts_result, 'prompts'):
                prompts = prompts_result.prompts
        except Exception as e:
            print(f"⚠️  No prompts available from MCP server {config.name}: {e}")
        
        return MCPConnection(
            client_context=client_context,
            stdio_session=stdio_session,
            session_context=session_context,
            mcp_session=mcp_session,
            tools=tools,
            prompts=prompts,
            config=config
        )
    
    def _get_server_key(self, config: MCPServerConfig) -> str:
        """Generate unique key for server configuration."""
        return f"{config.command}:{':'.join(config.args)}"
    
    async def get_tools(self, server_configs: List[MCPServerConfig], tool_tags: Optional[Set[str]] = None) -> List[Any]:
        """Get tools from specified servers, optionally filtered by tags.
        
        Args:
            server_configs: List of server configurations
            tool_tags: Optional set of tags to filter tools
            
        Returns:
            List of available tools
        """
        # Ensure servers are initialized
        await self.initialize_servers(server_configs)
        
        all_tools = []
        for config in server_configs:
            server_key = self._get_server_key(config)
            
            if server_key in self.tool_cache:
                server_tools = self.tool_cache[server_key]
                
                # Filter by requested tags if provided
                if tool_tags:
                    server_tools = self._filter_tools_by_tags(server_tools, tool_tags)
                
                all_tools.extend(server_tools)
        
        return all_tools
    
    def _filter_tools_by_tags(self, tools: List[Any], tags: Set[str]) -> List[Any]:
        """Filter tools by tags using hardcoded tool name mappings.
        
        NOTE: MCP Protocol doesn't support native tag filtering yet (GitHub Issue #522).
        This is a temporary workaround using hardcoded tool name mappings for Blocksworld.
        """
        if not tags:
            return tools
            
        # Hardcoded tool mappings for Blocksworld (temporary until MCP Protocol supports tags)
        blocksworld_tool_mappings = {
            "general": {"status", "rules"},  # General tools: status, rules
            "planning": {"status", "rules"},  # Planning tools: analyze current state and rules
            "verify": {"status", "rules", "validate_plan"},  # Verification tools: check status, rules, and verify plans
            "monitor": {"status", "rules"},  # Monitor tools: check status and rules
            "execution": {"pick_up", "unstack", "put_down", "stack"}  # Execution tools: perform actions
        }
        
        # Get allowed tool names for the requested tags
        allowed_tool_names = set()
        for tag in tags:
            if tag in blocksworld_tool_mappings:
                allowed_tool_names.update(blocksworld_tool_mappings[tag])
        
        if not allowed_tool_names:
            return tools
        
        # Filter tools by name
        filtered_tools = []
        for tool in tools:
            tool_name = getattr(tool, 'name', str(tool))
            if tool_name in allowed_tool_names:
                filtered_tools.append(tool)
        
        return filtered_tools
    
    def _load_all_servers_from_env(self) -> List[MCPServerConfig]:
        """Load all MCP server configurations from environment variables.
        
        Scans for environment variables matching the pattern:
        MCP_SERVER_<NAME>_COMMAND and MCP_SERVER_<NAME>_ARGS
        """
        servers = []
        
        # Find all MCP_SERVER_<NAME>_COMMAND environment variables
        for key, command in os.environ.items():
            if not (key.startswith("MCP_SERVER_") and key.endswith("_COMMAND")):
                continue
                
            # Extract server name (e.g. "BLOCKSWORLD" from "MCP_SERVER_BLOCKSWORLD_COMMAND")
            server_name = key[11:-8]  # Remove "MCP_SERVER_" prefix and "_COMMAND" suffix
            
            # Skip the legacy single-server config
            if server_name == "":  # This would be just "MCP_SERVER_COMMAND"
                continue
                
            # Get corresponding args
            args_key = f"MCP_SERVER_{server_name}_ARGS"
            args_str = os.getenv(args_key, "")
            args = [arg.strip() for arg in args_str.split()] if args_str else []
            
            server_config = MCPServerConfig(
                command=command,
                args=args,
                name=server_name.lower()
            )
            servers.append(server_config)
        
        return servers
    
    async def get_all_tools(self, allowed_tags: Optional[Set[str]] = None) -> List[Any]:
        """Get tools from all configured MCP servers, optionally filtered by tags.
        
        Args:
            allowed_tags: Set of tags to filter tools by. If None, all tools are returned.
            
        Returns:
            List of available tools
        """
        server_configs = self._load_all_servers_from_env()
        
        if not server_configs:
            # Fallback: try legacy single-server format
            server_configs = self._load_legacy_server_config()
        
        return await self.get_tools(server_configs, tool_tags=allowed_tags)
    
    def _load_legacy_server_config(self) -> List[MCPServerConfig]:
        """Load legacy single-server configuration for backwards compatibility."""
        mcp_command = os.getenv("MCP_SERVER_COMMAND", "")
        if not mcp_command:
            raise ValueError(
                "No MCP servers configured. Set environment variables in format "
                "MCP_SERVER_<NAME>_COMMAND or use legacy MCP_SERVER_COMMAND"
            )
        
        mcp_args_str = os.getenv("MCP_SERVER_ARGS", "")
        mcp_args = [arg.strip() for arg in mcp_args_str.split()] if mcp_args_str else []
        
        legacy_config = MCPServerConfig(
            command=mcp_command,
            args=mcp_args,
            name="default"
        )
        
        return [legacy_config]
    
    async def get_default_tools(self) -> List[Any]:
        """Get tools from default environment configuration (backwards compatibility)."""
        return await self.get_all_tools()
    
    async def get_prompts(self, server_configs: List[MCPServerConfig], prompt_tags: Optional[Set[str]] = None) -> List[Any]:
        """Get prompts from specified servers, optionally filtered by tags.
        
        Args:
            server_configs: List of server configurations
            prompt_tags: Optional set of tags to filter prompts
            
        Returns:
            List of available prompts
        """
        # Ensure servers are initialized
        await self.initialize_servers(server_configs)
        
        all_prompts = []
        for config in server_configs:
            server_key = self._get_server_key(config)
            
            if server_key in self.prompt_cache:
                server_prompts = self.prompt_cache[server_key]
                
                # Filter by requested tags if provided
                if prompt_tags:
                    server_prompts = self._filter_prompts_by_tags(server_prompts, prompt_tags)
                
                all_prompts.extend(server_prompts)
        
        return all_prompts
    
    def _filter_prompts_by_tags(self, prompts: List[Any], tags: Set[str]) -> List[Any]:
        """Filter prompts by tags based on prompt names or metadata."""
        if not tags:
            return prompts
            
        filtered_prompts = []
        for prompt in prompts:
            # Check if any tag appears in the prompt name (case insensitive)
            prompt_name = getattr(prompt, 'name', str(prompt)).lower()
            if any(tag.lower() in prompt_name for tag in tags):
                filtered_prompts.append(prompt)
        
        return filtered_prompts
    
    async def get_all_prompts(self, allowed_tags: Optional[Set[str]] = None) -> List[Any]:
        """Get prompts from all configured MCP servers, optionally filtered by tags.
        
        Args:
            allowed_tags: Set of tags to filter prompts by. If None, all prompts are returned.
            
        Returns:
            List of available prompts
        """
        server_configs = self._load_all_servers_from_env()
        
        if not server_configs:
            # Fallback: try legacy single-server format
            server_configs = self._load_legacy_server_config()
        
        return await self.get_prompts(server_configs, prompt_tags=allowed_tags)
    
    async def get_prompt_content(self, prompt_name: str, arguments: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get the content of a specific prompt by name with optional arguments.
        
        Args:
            prompt_name: Name of the prompt to retrieve
            arguments: Optional arguments to pass to the prompt
            
        Returns:
            Prompt content as string, or None if not found
        """
        for connection in self.connections.values():
            try:
                # Check if this server has the prompt
                for prompt in connection.prompts:
                    if getattr(prompt, 'name', '') == prompt_name:
                        # Get prompt content from server
                        result = await connection.mcp_session.get_prompt(
                            name=prompt_name, 
                            arguments=arguments or {}
                        )
                        if hasattr(result, 'messages') and result.messages:
                            # Combine all message content
                            content_parts = []
                            for message in result.messages:
                                if hasattr(message, 'content'):
                                    if hasattr(message.content, 'text'):
                                        content_parts.append(message.content.text)
                                    else:
                                        content_parts.append(str(message.content))
                            return '\n'.join(content_parts)
                        return str(result)
            except Exception as e:
                print(f"⚠️  Error retrieving prompt '{prompt_name}': {e}")
                continue
        
        return None
    
    async def close_all(self):
        """Close all MCP connections and cleanup resources."""
        async with self._lock:
            for server_key, connection in list(self.connections.items()):
                await self._close_connection_safely(server_key, connection)
            
            # Clear all caches and connections
            self.connections.clear()
            self.tool_cache.clear()
            self.prompt_cache.clear()
    
    async def _close_connection_safely(self, server_key: str, connection: MCPConnection):
        """Safely close a single MCP connection with proper error handling."""
        def _should_log_error(error: Exception) -> bool:
            """Check if error should be logged (ignore expected shutdown errors)."""
            error_str = str(error).lower()
            expected_errors = ["cancel scope", "generatorexit", "runtime error"]
            return not any(expected in error_str for expected in expected_errors)
        
        # Close session context
        if hasattr(connection, 'session_context') and connection.session_context:
            try:
                await connection.session_context.__aexit__(None, None, None)
            except Exception as e:
                if _should_log_error(e):
                    print(f"Warning: Error closing session context for {server_key}: {e}")
        
        # Close client context  
        if hasattr(connection, 'client_context') and connection.client_context:
            try:
                await connection.client_context.__aexit__(None, None, None)
            except Exception as e:
                if _should_log_error(e):
                    print(f"Warning: Error closing client context for {server_key}: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current MCP connections."""
        return {
            "active_servers": len(self.connections),
            "total_tools": sum(len(tools) for tools in self.tool_cache.values()),
            "total_prompts": sum(len(prompts) for prompts in self.prompt_cache.values()),
            "server_details": [
                {
                    "name": conn.config.name or conn.config.command,
                    "command": conn.config.command,
                    "tool_count": len(conn.tools),
                    "prompt_count": len(conn.prompts)
                }
                for conn in self.connections.values()
            ]
        }
    
    def __del__(self):
        """Destructor called when MCPManager is garbage collected."""
        try:
            # Only do synchronous cleanup to avoid hanging
            if hasattr(self, 'connections') and self.connections:
                # Clear references without async cleanup
                self.connections.clear()
                self.tool_cache.clear()
                if hasattr(self, 'prompt_cache'):
                    self.prompt_cache.clear()
        except Exception:
            pass  # Suppress all errors during destruction
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (mainly for testing)."""
        if cls._instance:
            try:
                # Try to cleanup before reset
                cls._instance.connections.clear()
                cls._instance.tool_cache.clear()
                if hasattr(cls._instance, 'prompt_cache'):
                    cls._instance.prompt_cache.clear()
            except Exception:
                pass
        cls._instance = None
        cls._initialized = False