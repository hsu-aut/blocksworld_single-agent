"""
Tool Event Classes for structured tool execution tracking.

This module provides clean object-oriented representations of tool events
instead of fragile string parsing.
"""

from dataclasses import dataclass
from typing import Optional, Any
import json


@dataclass
class ToolEvent:
    """Base class for all tool events."""
    tool_name: str
    
    def to_string(self) -> str:
        """Convert to string representation for backwards compatibility."""
        raise NotImplementedError
    
    @classmethod
    def from_string(cls, event_str: str) -> Optional['ToolEvent']:
        """Parse string representation back to object."""
        raise NotImplementedError


@dataclass
class ToolStartEvent(ToolEvent):
    """Tool execution started."""
    tool_input: Any
    
    def to_string(self) -> str:
        """Convert to string: TOOL_START:tool_name:input_json"""
        input_json = json.dumps(self.tool_input) if not isinstance(self.tool_input, str) else str(self.tool_input)
        return f"TOOL_START:{self.tool_name}:{input_json}"
    
    @classmethod
    def from_string(cls, event_str: str) -> Optional['ToolStartEvent']:
        """Parse: TOOL_START:tool_name:input_json"""
        if not event_str.startswith("TOOL_START:"):
            return None
        
        parts = event_str.split(":", 2)
        if len(parts) != 3:
            return None
            
        tool_name = parts[1]
        tool_input_str = parts[2].strip()
        
        # Try to parse as JSON, fallback to string
        try:
            tool_input = json.loads(tool_input_str)
        except (json.JSONDecodeError, ValueError):
            tool_input = tool_input_str
            
        return cls(tool_name=tool_name, tool_input=tool_input)


@dataclass
class ToolEndEvent(ToolEvent):
    """Tool execution completed successfully."""
    tool_output: Any
    
    def to_string(self) -> str:
        """Convert to string: TOOL_END:tool_name:output_json"""
        output_json = json.dumps(self.tool_output) if not isinstance(self.tool_output, str) else str(self.tool_output)
        return f"TOOL_END:{self.tool_name}:{output_json}"
    
    @classmethod
    def from_string(cls, event_str: str) -> Optional['ToolEndEvent']:
        """Parse: TOOL_END:tool_name:output_json"""
        if not event_str.startswith("TOOL_END:"):
            return None
        
        parts = event_str.split(":", 2)
        if len(parts) != 3:
            return None
            
        tool_name = parts[1]
        tool_output_str = parts[2].strip()
        
        # Try to parse as JSON, fallback to string
        try:
            tool_output = json.loads(tool_output_str)
        except (json.JSONDecodeError, ValueError):
            tool_output = tool_output_str
            
        return cls(tool_name=tool_name, tool_output=tool_output)


@dataclass
class ToolErrorEvent(ToolEvent):
    """Tool execution failed."""
    error_message: str
    
    def to_string(self) -> str:
        """Convert to string: TOOL_ERROR:tool_name:error_message"""
        return f"TOOL_ERROR:{self.tool_name}:{self.error_message}"
    
    @classmethod
    def from_string(cls, event_str: str) -> Optional['ToolErrorEvent']:
        """Parse: TOOL_ERROR:tool_name:error_message or TOOL_ERROR: error (Tool: name)"""
        if not event_str.startswith("TOOL_ERROR:"):
            return None
        
        # Handle new format: TOOL_ERROR:tool_name:error_message
        if event_str.count(":") >= 2:
            parts = event_str.split(":", 2)
            if len(parts) == 3:
                tool_name = parts[1]
                error_message = parts[2].strip()
                return cls(tool_name=tool_name, error_message=error_message)
        
        # Handle old format: TOOL_ERROR: error_message (Tool: tool_name)
        content = event_str[11:].strip()  # Remove "TOOL_ERROR: "
        tool_name = "unknown"
        error_message = content
        
        if "(Tool: " in content:
            tool_start = content.find("(Tool: ") + 7
            tool_end = content.find(")", tool_start)
            if tool_end > tool_start:
                tool_name = content[tool_start:tool_end]
                error_message = content[:content.find("(Tool:")].strip()
        
        return cls(tool_name=tool_name, error_message=error_message)


def parse_tool_event(event_str: str) -> Optional[ToolEvent]:
    """Parse any tool event string into appropriate object."""
    if event_str.startswith("TOOL_START:"):
        return ToolStartEvent.from_string(event_str)
    elif event_str.startswith("TOOL_END:"):
        return ToolEndEvent.from_string(event_str)
    elif event_str.startswith("TOOL_ERROR:"):
        return ToolErrorEvent.from_string(event_str)
    else:
        return None


@dataclass 
class ToolExecution:
    """Complete tool execution with all states."""
    name: str
    input: Any
    output: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if tool execution was successful."""
        return self.error is None
    
    @property
    def is_failed(self) -> bool:
        """Check if tool execution failed."""
        return self.error is not None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "input": self.input,
            "output": self.output,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ToolExecution':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            input=data["input"], 
            output=data.get("output"),
            error=data.get("error")
        )