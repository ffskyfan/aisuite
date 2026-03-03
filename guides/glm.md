# GLM-5

`aisuite` can route GLM-5 requests through Zhipu's official Python SDK.

## Install

```shell
pip install 'aisuite[glm]'
```

## Credentials

Set one of these environment variables:

```shell
export ZAI_API_KEY="your-api-key"
```

or

```shell
export ZHIPUAI_API_KEY="your-api-key"
```

## Usage

```python
import aisuite as ai

client = ai.Client({
    "glm": {
        "api_key": "your-api-key",
    }
})

response = client.chat.completions.create(
    model="glm:glm-5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "介绍一下你自己"},
    ],
)

print(response.choices[0].message.content)
```

You can also pass Zhipu SDK-specific arguments such as `thinking`, `tool_choice`, and `tool_stream`.
