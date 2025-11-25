"""MCP Configuration - Data structures for MCP server configuration.

This module defines the configuration structures used for MCP server
setup and management.
"""

from typing import List, Optional, Set, Any
from dataclasses import dataclass


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection.
    
    Attributes:
        command (str): Command to start the MCP server
        args (List[str]): Command line arguments for the server
        name (Optional[str]): Human-readable name for logging and identification
    """
    command: str
    args: List[str]
    name: Optional[str] = None


@dataclass
class MCPConnection:
    """Container for MCP connection objects.
    
    This stores all the async context managers and sessions needed
    for communication with an MCP server.
    """
    client_context: Any
    stdio_session: Any
    session_context: Any
    mcp_session: Any
    tools: List[Any]
    prompts: List[Any]  # Added support for MCP prompts
    config: MCPServerConfig