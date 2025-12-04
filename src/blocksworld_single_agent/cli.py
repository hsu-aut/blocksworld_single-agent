import asyncio
import argparse
import sys
from typing import Optional

from blocksworld_single_agent.core import ReActAgent, AVAILABLE_MODELS, MODELS_BY_PROVIDER_AND_SERIES
from blocksworld_single_agent.core.mcp import MCPManager
from blocksworld_single_agent.core.tool_events import parse_tool_event, ToolStartEvent, ToolEndEvent, ToolErrorEvent


async def test_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.1, execution_type: str = "react"):
    """Agent with interactive chat."""
    
    print(f"🤖 Starting {execution_type.upper()}: {model_name} (temp: {temperature})")
    
    # Create agent based on execution type
    if execution_type == "react":
        agent = ReActAgent(model_name, temperature)
        print(f"🔄 Single-Agent Planning  with Planning, Verification and Execution ")
    else:
        print(f"❌ Unknown execution type '{execution_type}'. Available: react")
        return
    
    print("=" * 50)
    
    try:
        print("⏳ Initializing agent...")
        success = await agent.initialize()
        
        if not success:
            print("❌ Failed to initialize agent")
            return
        
        print("✅ Agent ready!")
        print("💡 Type 'quit' or 'exit' to stop")
        print("💡 Type 'stats' to see token usage")
        print("=" * 50)
        
        # Universal interaction loop - same for all agent types
        while True:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                
                if user_input.lower() == "stats":
                    _print_stats(agent)
                    continue
                    
                if user_input.lower() == "help":
                    _print_help()
                    continue
                
                if not user_input:
                    continue
                
                # user prompt
                user_prompt= agent._get_default_system_prompt() + "\n "+ "User Input:"+ user_input
                
                # Send message and stream response with tool event handling
                await asyncio.wait_for(_stream_agent_response(agent, user_prompt), timeout=600)
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
            except asyncio.TimeoutError:
                print("⚠️ Streaming response timed out!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    finally:
        print("\n🧹 Cleaning up...")
        await agent.close()
        
        # Explicitly cleanup shared MCP resources
        try:
            mcp_manager = MCPManager()
            if mcp_manager.connections:
                print("🧹 Cleaning up shared MCP connections...")
                await mcp_manager.close_all()
        except Exception:
            # Suppress cleanup errors during shutdown
            pass
        
        print("✅ Done!")


async def _stream_agent_response(agent, user_input: str):
    """Stream agent response with clean tool event handling."""
    print("🤖 Agent: ", end="", flush=True)
    
    active_tools = set()  # Track active tools
    response_parts = []
    
    if not hasattr(agent, 'send_message'):
        print("❌ Agent does not support send_message interface")
        return
    
    try:
        async for chunk in agent.send_message(user_input):
            # Try to parse as tool event
            tool_event = parse_tool_event(chunk)
            
            if isinstance(tool_event, ToolStartEvent):
                active_tools.add(tool_event.tool_name)
                print(f"\n🔧 [Using {tool_event.tool_name}...]", end="", flush=True)
                
            elif isinstance(tool_event, ToolEndEvent):
                if tool_event.tool_name in active_tools:
                    active_tools.remove(tool_event.tool_name)
                    # Show abbreviated result
                    result_preview = str(tool_event.tool_output)[:50]
                    if len(str(tool_event.tool_output)) > 50:
                        result_preview += "..."
                    print(f"\n✅ [{tool_event.tool_name}: {result_preview}]\n🤖 ", end="", flush=True)
                    
            elif isinstance(tool_event, ToolErrorEvent):
                if tool_event.tool_name in active_tools:
                    active_tools.remove(tool_event.tool_name)
                    print(f"\n❌ [{tool_event.tool_name} failed: {tool_event.error_message}]\n🤖 ", end="", flush=True)
                    
            else:
                # Regular text chunk
                print(chunk, end="", flush=True)
                response_parts.append(chunk)
                
    except Exception as e:
        print(f"\n❌ Streaming error: {e}")
    
    print()  # New line after response
    
    # Clean up any remaining active tools (shouldn't happen, but safety)
    for tool_name in active_tools:
        print(f"⚠️  Warning: Tool '{tool_name}' was started but never completed")


def _print_stats(agent):
    """Print agent statistics."""
    stats = agent.get_stats()
    print(f"\n📊 Session Statistics:")
    print(f"   Tokens Used: {stats['total_tokens']:,}")
    print(f"   Estimated Cost: ${stats['total_cost']:.4f}")
    print(f"   Model: {stats['model_name']}")
    print(f"   Agent Type: {stats['agent_type']}")
    if 'graph_type' in stats:
        print(f"   Graph Type: {stats['graph_type']}")
    print()


def _print_help():
    """Print help information."""
    print(f"\n🆘 Available Commands:")
    print(f"   quit, exit, q    - Exit the chat")
    print(f"   stats           - Show token usage and costs")
    print(f"   help            - Show this help message")
    print(f"\n💡 Tips:")
    print(f"   - Use Ctrl+C to interrupt long responses")
    print(f"   - Tool executions are shown with 🔧 indicators")
    print(f"   - Successful tools show ✅, failed tools show ❌")
    print()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Quick Agent Tester")
    parser.add_argument(
        "--model", "-m", 
        default="gpt-4o-mini",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to use"
    )
    parser.add_argument(
        "--temp", "-t",
        type=float,
        default=0.1,
        help="Temperature (0.0-1.0)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--type", "-a",
        default="react",
        choices=["react", "multi-agent"],
        help="Execution type: react (direct agent) or multi-agent (planning workflow)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available models:")
        for model_name, config in AVAILABLE_MODELS.items():
            print(f"  {model_name:15} - {config['provider']:12} - {config['description']}")
        return
    
    # Validate temperature
    if not 0.0 <= args.temp <= 1.0:
        print("❌ Temperature must be between 0.0 and 1.0")
        return
    
    # Check if model is O-Series and warn about temperature override
    openai_models = MODELS_BY_PROVIDER_AND_SERIES.get("OpenAI", {})
    o_series_models = openai_models.get("O-Series", {})
    if args.model in o_series_models and args.temp != 1.0:
        print(f"⚠️  Warning: {args.model} is an O-Series model. Temperature will be automatically set to 1.0 (overriding {args.temp})")
        print()
    
    # Run interactive chat
    try:
        asyncio.run(test_agent(args.model, args.temp, args.type))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()