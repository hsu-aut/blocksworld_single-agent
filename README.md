# LLM MCP Client - Single Agent Framework

A modern multi-provider LLM client with Model Context Protocol (MCP) integration and CLI interface. This framework is designed for single-agent LLM integration with LangChain and MCP for planning and execution of different planning problems.

🚀 **Features:**
- 🤖 **Multi-Provider Support**: OpenAI, Anthropic, Google models
- 🔗 **MCP Integration**: External tool access via Model Context Protocol
- 💻 **CLI Interface**: Interactive command-line chat
- 📊 **Usage Tracking**: Real-time token and cost monitoring
- ⚡ **Agent Framework**: Built on LangGraph with ReAct pattern
- 🛠️ **Tool Events**: Rich tool execution feedback with structured events

Created for usage with the Blocksworld Simulation:
- [Simulation](https://github.com/hsu-aut/blocksworld_simulation) 
- [MCP Server](https://github.com/hsu-aut/llmstudy_mcp-server)

## Installation

Install dependencies using Poetry:

```bash
poetry install
```

## Configuration

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```bash
   # API Keys - Get these from respective providers
   GOOGLE_API_KEY=your_google_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here  
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # MCP Server Configuration
   MCP_SERVER_COMMAND=path_to_your_poetry_executable
   MCP_SERVER_ARGS=--directory /path/to/your/mcp/server run your-mcp-server
   ```

3. **Important**: Never commit your `.env` file - it's already in `.gitignore`


## Usage

### 💻 Command Line Interface
Run the interactive CLI chat:

```bash
poetry run llm-cli [options]

# Basic usage:
poetry run llm-cli                          # Uses default gpt-4o-mini with ReAct
poetry run llm-cli --model claude-3-haiku   # Use Claude model
poetry run llm-cli --temp 0.7               # Set temperature

# Utility commands:
poetry run llm-cli --list                   # List all available models
poetry run llm-cli --help                   # Show help

# Examples:
poetry run llm-cli --model gpt-4o --temp 0.2
poetry run llm-cli --model claude-3-sonnet --temp 0.5
```

**CLI Features:**
- 🤖 **Interactive Chat**: Continuous conversation with the agent
- 🔧 **Model Selection**: Choose from any supported model
- 🌡️ **Temperature Control**: Adjust model creativity
- 📊 **Usage Stats**: Type 'stats' to see token usage and costs
- 🛠️ **Tool Integration**: Full MCP tool support with visual feedback
- ⚡ **Streaming**: Real-time response streaming
- 🛑 **Interrupt Support**: Ctrl+C to stop generation

## Project Structure

```
src/llmstudy_mcp_client/
├── core/                           # Core components
│   ├── agents/                     # Agent implementations
│   │   ├── base.py                # Abstract AgentInterface
│   │   ├── react_agent.py         # ReAct agent implementation
│   │   └── simple_llm_agent.py    # Simple LLM agent
│   ├── mcp/                       # MCP integration
│   │   ├── manager.py             # MCP server management
│   │   ├── config.py              # MCP configuration
│   │   └── cleanup.py             # Resource cleanup
│   ├── models/                    # Model configurations
│   │   ├── factory.py             # Model factory
│   │   ├── config.py              # Model configurations
│   │   └── tracking.py            # Token/cost tracking
│   └── tool_events.py             # Tool execution events
└── cli.py                         # Command-line interface
```

## Available Models

The client supports multiple LLM providers with extensive model options:

### OpenAI
- **GPT-4.1 Series**: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- **GPT-4o Series**: `gpt-4o`, `gpt-4o-mini`, `gpt-4o-audio-preview`
- **GPT-4 Turbo**: `gpt-4-turbo`, `gpt-4-turbo-2024-04-09`
- **O-Series**: `o1`, `o1-mini`, `o1-preview`, `o3`, `o3-mini`
- **GPT-3.5**: `gpt-3.5-turbo`

### Anthropic
- **Claude 3.5**: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
- **Claude 3**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- **Short names**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

### Google
- **Gemini Pro**: `gemini-pro`, `gemini-1.5-pro`, `gemini-1.5-pro-002`
- **Gemini Flash**: `gemini-1.5-flash`, `gemini-1.5-flash-002`, `gemini-1.5-flash-8b`
- **Gemini 2.0**: `gemini-2.0-flash-exp`

## Key Features

### Agent Architecture
- **Abstract Interface**: `AgentInterface` provides a unified interface for all agent types
- **ReAct Implementation**: `ReActAgent` uses the Reasoning + Acting pattern
- **Tool Filtering**: Support for filtering tools by tags for agent specialization
- **Memory Management**: Conversation state persistence with checkpointers
- **Flexible Configuration**: Via `AgentConfig` dataclass with agent-specific parameters

### MCP Integration
- **Server Management**: Centralized `MCPManager` for efficient resource usage
- **Tool Loading**: Automatic tool discovery from MCP servers
- **Tag-based Filtering**: Fine-grained control over which tools the agent can access
- **Resource Cleanup**: Proper connection management and cleanup
- **Configuration**: Environment-based and programmatic setup

### Usage Tracking
- **Token Counting**: Real-time token usage monitoring per model
- **Cost Calculation**: Accurate cost tracking with current pricing
- **Statistics**: Session and agent-level statistics

### Tool Execution
- **Structured Events**: Rich feedback with `ToolStartEvent`, `ToolEndEvent`, `ToolErrorEvent`
- **Error Handling**: Comprehensive error reporting and handling
- **Streaming Support**: Real-time tool execution updates

## Advanced Usage Examples

### Simple Agent Usage
```python
from blocksworld_single_agent.core.agents import ReActAgent

# Create and initialize agent
agent = ReActAgent("gpt-4o-mini", temperature=0.1)
await agent.initialize()

# Send message and get streaming response
async for chunk in agent.send_message("Hello, how can you help me?"):
    print(chunk, end="")

# Clean up
await agent.close()
```

### Agent with Tool Filtering
```python
from blocksworld_single_agent.core.agents import ReActAgent
from blocksworld_single_agent.core.agents.base import AgentConfig

# Create agent with specific tool access
config = AgentConfig(
    model_name="gpt-4o-mini",
    temperature=0.1,
    system_prompt="You are a planning expert...",
    allowed_tool_tags={"planning", "simulation"},
    agent_specific_config={
        "max_iterations": 15,
        "early_stopping_method": "generate"
    }
)
agent = ReActAgent(config=config)
await agent.initialize()
```

## Architecture Patterns

- **Abstract Factory**: Model creation via factory pattern
- **Strategy Pattern**: Different agent implementations
- **Observer Pattern**: Token tracking callbacks
- **Singleton Pattern**: Shared MCP manager instance
- **Template Method**: Agent initialization sequence

## Requirements

- **Python**: 3.13+
- **Dependencies**: Managed via Poetry
- **API Keys**: OpenAI, Anthropic, Google (as needed)
- **MCP Server**: For external tool integration
