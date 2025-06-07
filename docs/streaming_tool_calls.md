# Streaming Tool Calls Guide

This guide explains how to properly handle tool calls in streaming mode with AISuite.

## Overview

When using streaming mode with tool calls, the tool call data is transmitted in chunks and needs special handling:

1. **Tool calls are sent incrementally** - The first chunk contains the tool call ID and function name, subsequent chunks contain pieces of the function arguments
2. **Arguments are sent as JSON fragments** - You need to accumulate the argument pieces until you have valid JSON
3. **Multiple tool calls can be streamed in parallel** - Each tool call has an index to identify which call the chunk belongs to

## Key Changes

### Framework Changes

- `ChoiceDelta` now includes a `tool_calls` field that contains **raw tool call deltas** (not converted)
- Providers pass through the original tool call delta format instead of converting them

### Provider Changes

- **DeepSeek Provider**: Passes raw `choice.delta.tool_calls` instead of converting them
- **OpenAI Provider**: Passes raw `choice.delta.tool_calls` instead of converting them
- **Other Providers**: Set `tool_calls=None` for compatibility

## Usage

### Basic Pattern

```python
from aisuite.utils.streaming_tool_calls import StreamingToolCallAccumulator

# Create accumulator
accumulator = StreamingToolCallAccumulator()

# Process streaming response
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
        # Process the complete tool calls...
        accumulator.clear()
```

### Complete Example

See `examples/streaming_tool_calls_example.py` for a full working example.

## StreamingToolCallAccumulator API

### Methods

- `add_chunk(tool_call_deltas)` - Add tool call deltas from a streaming chunk
- `has_complete_tool_calls()` - Check if any tool calls are ready for processing
- `get_complete_tool_calls()` - Get list of complete `ChatCompletionMessageToolCall` objects
- `clear()` - Clear all accumulated tool calls

### Tool Call Completion Logic

A tool call is considered complete when:
1. It has an ID
2. It has a function name  
3. It has function arguments that parse as valid JSON

## Important Notes

### Why Not Convert in Providers?

The original approach of using `_convert_tool_calls()` in providers was incorrect because:

1. **Partial data**: In streaming, tool call chunks often contain incomplete data
2. **JSON parsing errors**: Function arguments are sent as fragments and can't be parsed until complete
3. **Premature conversion**: Converting before accumulation leads to errors

### Correct Approach

1. **Providers**: Pass through raw tool call deltas without conversion
2. **Client code**: Use `StreamingToolCallAccumulator` to accumulate and detect completion
3. **Conversion**: Only convert to framework objects when tool calls are complete

## Migration Guide

If you were previously handling streaming tool calls manually:

### Before (Incorrect)
```python
# This would fail with partial data
for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        # This could fail with incomplete JSON
        tool_calls = convert_tool_calls(chunk.choices[0].delta.tool_calls)
```

### After (Correct)
```python
accumulator = StreamingToolCallAccumulator()

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        accumulator.add_chunk(chunk.choices[0].delta.tool_calls)
        
    if accumulator.has_complete_tool_calls():
        complete_tool_calls = accumulator.get_complete_tool_calls()
        # Process complete tool calls
        accumulator.clear()
```

## Supported Providers

- ✅ **DeepSeek Provider** - Full streaming tool calls support
- ✅ **OpenAI Provider** - Full streaming tool calls support  
- ⚠️ **Anthropic Provider** - Framework compatible, different tool call handling
- ⚠️ **Gemini Provider** - Framework compatible, no streaming tool calls yet
- ⚠️ **Other Providers** - Framework compatible, may not support tool calls

## Troubleshooting

### Common Issues

1. **JSON Parse Errors**: Make sure to use the accumulator and only process complete tool calls
2. **Missing Tool Calls**: Check that your provider supports streaming tool calls
3. **Incomplete Arguments**: Wait for `has_complete_tool_calls()` to return `True`

### Debug Tips

```python
# Add debug logging
accumulator = StreamingToolCallAccumulator()

for chunk in stream:
    if chunk.choices[0].delta.tool_calls:
        print(f"Tool call chunk: {chunk.choices[0].delta.tool_calls}")
        accumulator.add_chunk(chunk.choices[0].delta.tool_calls)
        print(f"Current state: {accumulator.tool_calls}")
```
