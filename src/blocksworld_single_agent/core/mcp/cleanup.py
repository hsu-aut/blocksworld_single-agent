"""MCP Cleanup utilities for proper shutdown of MCP resources.

This module provides utilities for graceful shutdown of MCP-related resources
like the MCP Manager, ensuring proper cleanup when applications exit.

Usage:
    import atexit
    from .cleanup import register_mcp_cleanup_handlers
    
    # Register cleanup handlers
    register_mcp_cleanup_handlers()
    
    # Or manually cleanup
    await cleanup_shared_mcp_resources()
"""

import atexit
import asyncio


async def cleanup_shared_mcp_resources():
    """Clean up all shared MCP resources like MCP Manager."""
    try:
        # Import here to avoid circular imports
        from .manager import MCPManager
        
        manager = MCPManager()
        if manager.connections:
            await manager.close_all()
    except Exception as e:
        # Suppress common shutdown errors that are expected
        if not any(err in str(e).lower() for err in ["cancel scope", "generatorexit", "runtime error"]):
            print(f"Warning: Error during MCP cleanup: {e}")


def register_mcp_cleanup_handlers():
    """Register handlers to cleanup shared MCP resources on application exit."""
    
    def sync_cleanup():
        """Synchronous wrapper for async MCP cleanup."""
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we can't block - just return
                    # The resources will be cleaned up when the loop closes
                    return
                else:
                    # Loop exists but not running - we can use it
                    loop.run_until_complete(cleanup_shared_mcp_resources())
                    return
            except RuntimeError:
                # No event loop in this thread
                pass
            
            # Create new event loop for cleanup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(cleanup_shared_mcp_resources())
            finally:
                # Always close the loop we created
                loop.close()
                
        except Exception as e:
            # Only print unexpected errors
            if not any(err in str(e).lower() for err in ["cancel scope", "generatorexit", "runtime error", "coroutine"]):
                print(f"Warning: Could not cleanup MCP resources: {e}")
    
    # Register cleanup on normal exit
    atexit.register(sync_cleanup)