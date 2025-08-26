# Vercel AI Gateway Provider

The Vercel AI Gateway provider allows you to access multiple AI models through Vercel's unified AI Gateway API. This provider uses OpenAI-compatible endpoints and supports various models from different providers like OpenAI, Anthropic, Google, and more.

## What is Vercel AI Gateway?

Vercel AI Gateway is a unified API that provides access to multiple AI providers through a single OpenAI-compatible interface. It offers:

- **Multi-provider support**: Access models from OpenAI, Anthropic, Google, and more
- **Provider routing**: Automatic fallback between providers for reliability
- **Unified billing**: Single billing through your Vercel account
- **Observability**: Built-in monitoring and analytics
- **Rate limiting**: Centralized rate limit management

## Features

- ✅ Chat completions (sync and async)
- ✅ Streaming responses
- ✅ Tool/function calling
- ✅ Multiple model providers through single gateway
- ✅ Image and PDF attachments (if supported by underlying model)
- ✅ Provider routing and fallback options

## Installation

The Vercel provider uses the OpenAI client library:

```bash
pip install aisuite[vercel]
# or
pip install aisuite[openai]  # since vercel provider depends on openai
```

## Configuration

### API Key

You need a Vercel AI Gateway API key. Set it as an environment variable:

```bash
export AI_GATEWAY_API_KEY="your-api-key-here"
```

Or pass it directly in the configuration:

```python
import aisuite as ai

client = ai.Client({
    "vercel": {
        "api_key": "your-api-key-here"
    }
})
```

### Alternative Authentication

You can also use Vercel OIDC tokens:

```bash
export VERCEL_OIDC_TOKEN="your-oidc-token"
```

## Usage

### Basic Chat Completion

```python
import aisuite as ai

client = ai.Client({
    "vercel": {
        "api_key": "your-api-key-here"
    }
})

# Use models in the format: vercel:provider/model
response = await client.chat.completions.create(
    model="vercel:anthropic/claude-sonnet-4",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

### Available Models

The Vercel AI Gateway supports models from various providers:

- **OpenAI**: `vercel:openai/gpt-4o`, `vercel:openai/gpt-4o-mini`
- **Anthropic**: `vercel:anthropic/claude-sonnet-4`, `vercel:anthropic/claude-haiku-4`
- **Google**: `vercel:google/gemini-pro`, `vercel:google/gemini-flash`
- **And many more...**

You can list available models using the gateway's API:

```python
# This would require direct API call to /v1/models endpoint
# The exact model names depend on your gateway configuration
```

### Streaming

```python
stream = await client.chat.completions.create(
    model="vercel:openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

async for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Tool/Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

response = await client.chat.completions.create(
    model="vercel:openai/gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function called: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

### Provider Options

You can specify provider routing preferences:

```python
response = await client.chat.completions.create(
    model="vercel:anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Hello"}],
    # This would require extending the provider to support providerOptions
    extra_body={
        "providerOptions": {
            "gateway": {
                "order": ["vertex", "anthropic"]  # Try Vertex AI first, then Anthropic
            }
        }
    }
)
```

### Image Analysis

```python
import base64

# Read and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response = await client.chat.completions.create(
    model="vercel:anthropic/claude-sonnet-4",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }
    ]
)
```

## Error Handling

```python
from aisuite.provider import LLMError

try:
    response = await client.chat.completions.create(
        model="vercel:invalid/model",
        messages=[{"role": "user", "content": "Hello"}]
    )
except LLMError as e:
    print(f"LLM Error: {e}")
except Exception as e:
    print(f"Other error: {e}")
```

## Environment Variables

- `AI_GATEWAY_API_KEY`: Your Vercel AI Gateway API key
- `VERCEL_OIDC_TOKEN`: Alternative authentication using OIDC token

## Notes

1. The Vercel AI Gateway acts as a proxy to multiple AI providers
2. Model availability depends on your gateway configuration
3. Pricing and rate limits are managed through your Vercel account
4. The provider uses OpenAI-compatible API format internally
5. All features supported by the underlying models should work through the gateway

## Troubleshooting

### Common Issues

1. **Authentication Error**: Make sure your API key is valid and has proper permissions
2. **Model Not Found**: Verify the model name format and availability in your gateway
3. **Rate Limiting**: Check your Vercel account for rate limit settings
4. **Network Issues**: Ensure you can reach `https://ai-gateway.vercel.sh/v1`

### Debug Mode

Enable debug logging to see detailed request/response information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
