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
import aisuite as ai
client = ai.Client()

provider = "deepseek"
model_id = "deepseek-v4-flash"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What’s the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

## DeepSeek V4 Models

The current DeepSeek API model IDs are:

- `deepseek-v4-flash`: fast, economical default.
- `deepseek-v4-pro`: higher quality model for harder reasoning, coding, and agent work.

DeepSeek V4 defaults to thinking mode. To explicitly control it through `aisuite`, pass `thinking`; the provider will forward it via the OpenAI SDK `extra_body` field:

```python
response = client.chat.completions.create(
    model="deepseek:deepseek-v4-flash",
    messages=messages,
    thinking={"type": "disabled"},
)
```

For thinking mode, use `reasoning_effort` with `high` or `max`:

```python
response = client.chat.completions.create(
    model="deepseek:deepseek-v4-pro",
    messages=messages,
    thinking={"type": "enabled"},
    reasoning_effort="high",
)
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
