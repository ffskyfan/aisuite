#!/usr/bin/env python3
"""
Example of how to properly handle streaming tool calls with DeepSeek provider.

This example demonstrates:
1. How to accumulate tool call chunks during streaming
2. How to detect when tool calls are complete
3. How to execute functions and continue the conversation
"""

import asyncio
import os
import json
from aisuite.providers.deepseek_provider import DeepseekProvider
from aisuite.utils.streaming_tool_calls import StreamingToolCallAccumulator


def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Mock weather function for demonstration."""
    weather_data = {
        "tokyo": {"temp": 10, "condition": "cloudy"},
        "san francisco": {"temp": 72, "condition": "sunny"},
        "paris": {"temp": 22, "condition": "rainy"},
    }
    
    location_lower = location.lower()
    if location_lower in weather_data:
        data = weather_data[location_lower]
        return json.dumps({
            "location": location,
            "temperature": data["temp"],
            "unit": unit,
            "condition": data["condition"]
        })
    else:
        return json.dumps({
            "location": location,
            "temperature": "unknown",
            "condition": "unknown"
        })


def calculate(expression: str) -> str:
    """Mock calculation function for demonstration."""
    try:
        # Simple evaluation - in production, use a safer approach
        result = eval(expression)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)})


# Available functions mapping
AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# Tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


async def streaming_chat_with_tools():
    """Demonstrate streaming chat with tool calls."""
    
    # Check for API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Please set DEEPSEEK_API_KEY environment variable")
        return
    
    provider = DeepseekProvider()
    
    # Initial conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to weather and calculation tools."},
        {"role": "user", "content": "What's the weather in Tokyo and what's 15 * 23?"}
    ]
    
    print("User: What's the weather in Tokyo and what's 15 * 23?")
    print("Assistant: ", end="", flush=True)
    
    # Create tool call accumulator
    accumulator = StreamingToolCallAccumulator()
    assistant_content = ""
    
    # First API call with streaming
    response = await provider.chat_completions_create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,
        stream=True
    )
    
    # Process streaming response
    async for chunk in response:
        choice = chunk.choices[0]
        delta = choice.delta
        
        # Handle content
        if delta.content:
            print(delta.content, end="", flush=True)
            assistant_content += delta.content
        
        # Accumulate tool calls
        if delta.tool_calls:
            accumulator.add_chunk(delta.tool_calls)
        
        # Check if stream is finished
        if choice.finish_reason:
            break
    
    print()  # New line after streaming content
    
    # Check if we have complete tool calls
    if accumulator.has_complete_tool_calls():
        complete_tool_calls = accumulator.get_complete_tool_calls()
        
        print(f"\n🔧 Executing {len(complete_tool_calls)} tool call(s)...")
        
        # Add assistant message with tool calls to conversation
        # Convert our tool calls back to the format expected by the API
        tool_calls_for_api = []
        for tool_call in complete_tool_calls:
            tool_calls_for_api.append({
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
        
        messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": tool_calls_for_api
        })
        
        # Execute each tool call
        for tool_call in complete_tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"  📞 Calling {function_name}({function_args})")
            
            if function_name in AVAILABLE_FUNCTIONS:
                function_to_call = AVAILABLE_FUNCTIONS[function_name]
                function_response = function_to_call(**function_args)
                
                # Add tool response to conversation
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })
                
                print(f"  ✅ Result: {function_response}")
            else:
                print(f"  ❌ Unknown function: {function_name}")
        
        print("\n🤖 Assistant (after tool calls): ", end="", flush=True)
        
        # Make follow-up API call to get final response
        response = await provider.chat_completions_create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )
        
        # Stream the final response
        async for chunk in response:
            choice = chunk.choices[0]
            if choice.delta.content:
                print(choice.delta.content, end="", flush=True)
        
        print()  # Final new line
    
    else:
        print("\n✅ No tool calls needed.")


async def main():
    """Main function."""
    print("🚀 Streaming Tool Calls Example with DeepSeek")
    print("=" * 50)
    
    try:
        await streaming_chat_with_tools()
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())
