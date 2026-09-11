# DeepSeek

To use DeepSeek with `aisuite`, you’ll need an [DeepSeek account](https://platform.deepseek.com). After logging in, go to the [API Keys](https://platform.deepseek.com/api_keys) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

## Create a Chat Completion

DeepSeek uses an API format compatible with OpenAI, so the `openai` Python client is required.

Install the `openai` Python client:

Example with pip:
```shell
pip install openai
```

Example with poetry:
```shell
poetry add openai
```

In your code:
```python
import asyncio
import aisuite as ai

async def main():
    client = ai.Client()
    try:
        response = await client.chat.completions.create(
            model="deepseek:deepseek-flash",
            messages=[{"role": "user", "content": "Say hello."}],
        )
        print(response.choices[0].message.content)
    finally:
        await client.aclose()

asyncio.run(main())
```

## DeepSeek V4.1 Flash

Use `deepseek-flash` for DeepSeek V4.1 Flash, including native vision. The retired
`deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` names already route to it.
The generic provider still accepts other API model IDs, but only explicitly
known visual models preserve images; it does not infer capabilities from names.

DeepSeek defaults to thinking mode. To explicitly control it through `aisuite`, pass `thinking`; the provider will forward it via the OpenAI SDK `extra_body` field:

The following snippets run inside an async function with an initialized client.

```python
response = await client.chat.completions.create(
    model="deepseek:deepseek-flash",
    messages=messages,
    thinking={"type": "disabled"},
)
```

For thinking mode, use `reasoning_effort` with `low`, `high`, or `max`:

```python
response = await client.chat.completions.create(
    model="deepseek:deepseek-flash",
    messages=messages,
    thinking={"type": "enabled"},
    reasoning_effort="high",
)
```

User images use OpenAI-compatible `image_url` content blocks. Tool-result images
are projected to a user message after the complete tool-result group; tool IDs,
text, and canonical history are preserved. Thinking history is replayed in
`reasoning_content` when tools are used.

Sources (checked 2026-09-11): [models](https://api-docs.deepseek.com/quick_start/pricing/),
[vision](https://api-docs.deepseek.com/guides/vision/),
[thinking](https://api-docs.deepseek.com/guides/thinking_mode/).

## Reasoning history

New responses store the original reasoning text only in
`reasoning_content.thinking`, with `raw_data=None`. This also preserves an empty
string for tool-call turns. The provider converts that text into a single
`reasoning_content` string when building the API request.

Existing histories with versioned `raw_data.payload.reasoning_content` or legacy
`raw_data.reasoning_content` remain readable without rewriting the stored data.
If old copies disagree, replay retains its existing precedence: the versioned
payload, then the legacy raw field, then canonical `thinking`.

Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
