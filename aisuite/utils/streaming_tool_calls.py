"""
Utilities for handling streaming tool calls.

In streaming mode, tool calls are sent in chunks and need to be accumulated
before they can be processed. This module provides utilities to help with that.
"""

from typing import List, Dict, Any, Optional
import json
from aisuite.framework.message import ChatCompletionMessageToolCall, Function


class StreamingToolCallAccumulator:
    """
    Accumulates tool call chunks from streaming responses and converts them
    to complete tool calls when ready.
    """
    
    def __init__(self):
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
    
    def add_chunk(self, tool_call_deltas: List[Any]) -> None:
        """
        Add a chunk of tool call deltas to the accumulator.
        
        Args:
            tool_call_deltas: List of tool call delta objects from streaming response
        """
        if not tool_call_deltas:
            return
            
        for delta in tool_call_deltas:
            index = getattr(delta, 'index', 0)
            
            # Initialize tool call if not exists
            if index not in self.tool_calls:
                self.tool_calls[index] = {
                    "id": "",
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": ""
                    }
                }
            
            tool_call = self.tool_calls[index]
            
            # Accumulate id
            if hasattr(delta, 'id') and delta.id:
                tool_call["id"] += delta.id
            
            # Accumulate function name
            if hasattr(delta, 'function') and delta.function:
                if hasattr(delta.function, 'name') and delta.function.name:
                    tool_call["function"]["name"] += delta.function.name
                
                # Accumulate function arguments
                if hasattr(delta.function, 'arguments') and delta.function.arguments:
                    tool_call["function"]["arguments"] += delta.function.arguments
            
            # Set type if provided
            if hasattr(delta, 'type') and delta.type:
                tool_call["type"] = delta.type
    
    def get_complete_tool_calls(self) -> List[ChatCompletionMessageToolCall]:
        """
        Get complete tool calls that have valid JSON arguments.
        
        Returns:
            List of complete ChatCompletionMessageToolCall objects
        """
        complete_calls = []
        
        for tool_call_data in self.tool_calls.values():
            # Check if we have all required fields
            if (tool_call_data["id"] and 
                tool_call_data["function"]["name"] and 
                tool_call_data["function"]["arguments"]):
                
                try:
                    # Try to parse arguments as JSON to ensure completeness
                    json.loads(tool_call_data["function"]["arguments"])
                    
                    # Create the tool call object
                    function = Function(
                        name=tool_call_data["function"]["name"],
                        arguments=tool_call_data["function"]["arguments"]
                    )
                    
                    tool_call = ChatCompletionMessageToolCall(
                        id=tool_call_data["id"],
                        function=function,
                        type=tool_call_data["type"]
                    )
                    
                    complete_calls.append(tool_call)
                    
                except json.JSONDecodeError:
                    # Arguments are not complete yet
                    continue
        
        return complete_calls
    
    def has_complete_tool_calls(self) -> bool:
        """
        Check if there are any complete tool calls ready for processing.
        
        Returns:
            True if there are complete tool calls, False otherwise
        """
        return len(self.get_complete_tool_calls()) > 0
    
    def clear(self) -> None:
        """Clear all accumulated tool calls."""
        self.tool_calls.clear()


def accumulate_streaming_tool_calls(chunks: List[Any]) -> List[ChatCompletionMessageToolCall]:
    """
    Convenience function to accumulate tool calls from a list of streaming chunks.
    
    Args:
        chunks: List of streaming response chunks
        
    Returns:
        List of complete ChatCompletionMessageToolCall objects
    """
    accumulator = StreamingToolCallAccumulator()
    
    for chunk in chunks:
        if (hasattr(chunk, 'choices') and chunk.choices and 
            hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta and
            hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls):
            
            accumulator.add_chunk(chunk.choices[0].delta.tool_calls)
    
    return accumulator.get_complete_tool_calls()


# Example usage:
"""
# In your streaming handler:
accumulator = StreamingToolCallAccumulator()

async for chunk in stream:
    # Handle content
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
    
    # Accumulate tool calls
    if chunk.choices[0].delta.tool_calls:
        accumulator.add_chunk(chunk.choices[0].delta.tool_calls)
    
    # Check if we have complete tool calls
    if accumulator.has_complete_tool_calls():
        complete_tool_calls = accumulator.get_complete_tool_calls()
        # Process the complete tool calls
        for tool_call in complete_tool_calls:
            # Execute the function
            result = execute_function(tool_call)
            # Add to messages for next API call
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool", 
                "name": tool_call.function.name,
                "content": result
            })
        accumulator.clear()
"""
