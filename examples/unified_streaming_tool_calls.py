#!/usr/bin/env python3
"""
Example demonstrating unified tool call handling for both streaming and non-streaming modes.

This example shows that with the provider-level accumulation approach, 
the upper-level code can handle tool calls identically regardless of streaming mode.
"""

import asyncio
import os
import json
from aisuite.providers.deepseek_provider import DeepseekProvider


def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Mock weather function."""
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
    """Mock calculation function."""
    try:
        result = eval(expression)  # In production, use a safer approach
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"expression": expression, "error": str(e)})


# Available functions
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
                    "location": {"type": "string", "description": "The city name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
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
                    "expression": {"type": "string", "description": "Mathematical expression"}
                },
                "required": ["expression"]
            }
        }
    }
]


async def handle_tool_calls(tool_calls, messages):
    """
    Handle tool calls - this function works identically for streaming and non-streaming!
    
    Args:
        tool_calls: List of ChatCompletionMessageToolCall objects (unified format)
        messages: Conversation messages list to append results to
    """
    if not tool_calls:
        return
    
    print(f"\n🔧 Executing {len(tool_calls)} tool call(s)...")
    
    for tool_call in tool_calls:
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


async def chat_with_streaming(provider, messages):
    """Chat with streaming mode."""
    print("🌊 Streaming Mode:")
    print("Assistant: ", end="", flush=True)
    
    assistant_content = ""
    tool_calls_found = []
    
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
        
        # Handle content - streams in real-time
        if delta.content:
            print(delta.content, end="", flush=True)
            assistant_content += delta.content
        
        # Handle tool calls - unified format, same as non-streaming!
        if delta.tool_calls:
            tool_calls_found.extend(delta.tool_calls)
        
        if choice.finish_reason:
            break
    
    print()  # New line after streaming
    
    # Add assistant message to conversation
    if tool_calls_found:
        # Convert tool calls to API format for message history
        tool_calls_for_api = []
        for tool_call in tool_calls_found:
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
        
        # Handle tool calls using the same function as non-streaming!
        await handle_tool_calls(tool_calls_found, messages)
        
        # Get final response
        print("\n🤖 Assistant (after tool calls): ", end="", flush=True)
        
        response = await provider.chat_completions_create(
            model="deepseek-chat",
            messages=messages,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        
        print()
    else:
        messages.append({"role": "assistant", "content": assistant_content})


async def chat_without_streaming(provider, messages):
    """Chat without streaming mode."""
    print("\n📄 Non-Streaming Mode:")
    
    # API call without streaming
    response = await provider.chat_completions_create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,
        stream=False
    )
    
    choice = response.choices[0]
    message = choice.message
    
    print(f"Assistant: {message.content}")
    
    # Add assistant message to conversation
    if message.tool_calls:
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        })
        
        # Handle tool calls using the SAME function as streaming!
        await handle_tool_calls(message.tool_calls, messages)
        
        # Get final response
        response = await provider.chat_completions_create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )
        
        final_message = response.choices[0].message
        print(f"Assistant (after tool calls): {final_message.content}")
    else:
        messages.append({"role": "assistant", "content": message.content})


async def main():
    """Main demonstration."""
    print("🚀 Unified Tool Call Handling Demo")
    print("=" * 50)
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Please set DEEPSEEK_API_KEY environment variable")
        return
    
    provider = DeepseekProvider()
    
    # Test query that will trigger tool calls
    user_query = "What's the weather in Tokyo and what's 15 * 23?"
    
    print(f"User: {user_query}")
    
    # Test 1: Streaming mode
    messages_streaming = [
        {"role": "system", "content": "You are a helpful assistant with access to weather and calculation tools."},
        {"role": "user", "content": user_query}
    ]
    
    try:
        await chat_with_streaming(provider, messages_streaming)
    except Exception as e:
        print(f"Streaming mode error: {e}")
    
    # Test 2: Non-streaming mode (same logic!)
    messages_non_streaming = [
        {"role": "system", "content": "You are a helpful assistant with access to weather and calculation tools."},
        {"role": "user", "content": user_query}
    ]
    
    try:
        await chat_without_streaming(provider, messages_non_streaming)
    except Exception as e:
        print(f"Non-streaming mode error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("\n🎯 Key points demonstrated:")
    print("  • Same tool call handling logic for both modes")
    print("  • Unified format from provider layer")
    print("  • Real-time content streaming")
    print("  • Proper tool call accumulation")


if __name__ == "__main__":
    asyncio.run(main())
