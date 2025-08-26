# CloseAI Provider

CloseAI provider for aisuite, enabling access to multiple AI models through CloseAI's enterprise-grade OpenAI proxy service.

## Overview

CloseAI is an enterprise-level commercial OpenAI proxy platform that provides:
- Full OpenAI API compatibility
- Multi-model aggregation interface
- Automatic protocol conversion between different AI providers
- Extended timeout support for reasoning models (up to 20 minutes)
- Load balancing across multiple accounts
- Access to models from OpenAI, Anthropic, Google Gemini, and more

## Configuration

### Environment Variables

Set your CloseAI API key as an environment variable:

```bash
export CLOSEAI_API_KEY="sk-your-closeai-api-key"
```

### Client Configuration

```python
import aisuite as ai

client = ai.Client({
    "closeai": {
        "api_key": "sk-your-closeai-api-key",  # Optional if set as environment variable
        "base_url": "https://api.openai-proxy.org/v1"  # Optional, this is the default
    }
})
```

## Usage Examples

### Basic Chat Completion

```python
import aisuite as ai

client = ai.Client({"closeai": {"api_key": "sk-your-closeai-api-key"}})

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

response = client.chat.completions.create(
    model="closeai:gpt-4",
    messages=messages,
    max_tokens=150
)

print(response.choices[0].message.content)
```

### Streaming Response

```python
import aisuite as ai

client = ai.Client({"closeai": {"api_key": "sk-your-closeai-api-key"}})

messages = [
    {"role": "user", "content": "Write a short story about a robot."},
]

stream = client.chat.completions.create(
    model="closeai:gpt-4",
    messages=messages,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Using Different Model Providers

CloseAI's multi-model aggregation allows you to access models from different providers:

```python
# OpenAI models
response = client.chat.completions.create(
    model="closeai:gpt-4",
    messages=messages
)

# Anthropic models (automatically converted)
response = client.chat.completions.create(
    model="closeai:claude-3-sonnet-20240229",
    messages=messages
)

# Google Gemini models (automatically converted)
response = client.chat.completions.create(
    model="closeai:gemini-pro",
    messages=messages
)
```

### Reasoning Models

CloseAI supports OpenAI's reasoning models with extended timeout support:

```python
# Using o1 series reasoning models
response = client.chat.completions.create(
    model="closeai:o1-preview",
    messages=[
        {"role": "user", "content": "Solve this complex math problem step by step: ..."}
    ],
    reasoning_effort="high"  # CloseAI automatically handles this
)

# Access reasoning content if available
if response.choices[0].message.reasoning_content:
    print("Reasoning:", response.choices[0].message.reasoning_content.thinking)
```

### Tool Calls (Function Calling)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="closeai:gpt-4",
    messages=[
        {"role": "user", "content": "What's the weather like in Tokyo?"}
    ],
    tools=tools,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Function: {tool_call.function.name}")
        print(f"Arguments: {tool_call.function.arguments}")
```

## Features

### Supported Features

- ✅ Chat completions (non-streaming and streaming)
- ✅ Function calling / Tool use
- ✅ Reasoning models (o1, o3 series)
- ✅ Multi-model access (OpenAI, Anthropic, Gemini, etc.)
- ✅ Automatic protocol conversion
- ✅ Extended timeout for complex reasoning tasks
- ✅ Load balancing across multiple accounts

### Limitations

- ❌ Stateful interfaces (file uploads, fine-tuning, assistants)
- ❌ Response API stateful usage (cannot pass previous response IDs)

### Model Support

CloseAI supports a wide range of models through its aggregation interface:

**OpenAI Models:**
- GPT-4 series (gpt-4, gpt-4-turbo, gpt-4o, etc.)
- GPT-3.5 series (gpt-3.5-turbo, etc.)
- Reasoning models (o1-preview, o1-mini, o3-pro, etc.)

**Anthropic Models:**
- Claude 3 series (claude-3-opus, claude-3-sonnet, claude-3-haiku)
- Claude 3.5 series (claude-3-5-sonnet, etc.)

**Google Models:**
- Gemini series (gemini-pro, gemini-pro-vision, etc.)

**Other Models:**
- Various other models supported by CloseAI's platform

## Advanced Configuration

### Custom Base URL

If you need to use a different CloseAI endpoint:

```python
client = ai.Client({
    "closeai": {
        "api_key": "sk-your-closeai-api-key",
        "base_url": "https://your-custom-closeai-endpoint.com/v1"
    }
})
```

### Timeout Configuration

For reasoning models that may take longer to respond:

```python
import httpx

client = ai.Client({
    "closeai": {
        "api_key": "sk-your-closeai-api-key",
        "timeout": httpx.Timeout(60.0)  # 60 second timeout
    }
})
```

## Error Handling

```python
try:
    response = client.chat.completions.create(
        model="closeai:gpt-4",
        messages=messages
    )
except Exception as e:
    print(f"Error: {e}")
```

## Best Practices

1. **Model Selection**: Use CloseAI's multi-model aggregation to access the best model for your use case
2. **Reasoning Models**: For complex reasoning tasks, use o1/o3 series models with appropriate reasoning_effort
3. **Timeout Handling**: Set appropriate timeouts for reasoning models that may take longer to respond
4. **Error Handling**: Always implement proper error handling for API calls
5. **Rate Limiting**: Be aware of CloseAI's rate limiting policies

## Notes

- CloseAI automatically handles protocol conversion between different model providers
- The platform uses load balancing across multiple accounts for better reliability
- Reasoning models support extended timeouts up to 20 minutes (vs 5 minutes for standard ChatCompletion)
- All OpenAI parameters and features are supported through CloseAI's compatible interface

**Note**: CloseAI is a third-party service. Please refer to their official documentation and terms of service for the most up-to-date information about features, pricing, and usage policies.
