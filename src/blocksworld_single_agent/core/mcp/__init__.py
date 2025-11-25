"""
MCP Package - Model Context Protocol management and utilities.

This package contains all MCP-related functionality including:
- MCP server management and connection pooling
- Configuration loading from environment
- Resource cleanup and lifecycle management

The package provides a clean separation between MCP concerns and
the rest of the application.
"""

from .manager import MCPManager
from .config import MCPServerConfig
from .cleanup import cleanup_shared_mcp_resources, register_mcp_cleanup_handlers

__all__ = [
    # Core MCP management
    "MCPManager",
    "MCPServerConfig",
    
    # Cleanup utilities
    "cleanup_shared_mcp_resources",
    "register_mcp_cleanup_handlers"
]