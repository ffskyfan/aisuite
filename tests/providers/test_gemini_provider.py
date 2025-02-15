from unittest.mock import MagicMock, patch
import pytest
import os
from aisuite.providers.gemini_provider import GeminiProvider
from aisuite.framework.message import Message
from dotenv import load_dotenv

@pytest.fixture(autouse=True)
def set_api_key_env_var():
    """Fixture to load environment variables for tests."""
    load_dotenv()
    os.environ["IS_TESTING"] = "True"


@pytest.mark.asyncio
async def test_gemini_provider_non_streaming():
    """Test non-streaming chat completions."""

    user_message = "Hello!"
    messages = [Message(role="user", content=user_message, tool_calls=None, refusal=None)]
    model = "gemini-2.0-flash"
    response_text = "Hi there!"

    provider = GeminiProvider()
    response = await provider.chat_completions_create(
        messages=messages, model=model, response_text=response_text
    )
    assert response.choices[0].message == response_text
    assert response.choices[0].finish_reason is None
    assert response.metadata["model"] == "mocked-model"


@pytest.mark.asyncio
async def test_gemini_provider_streaming():
    """Test streaming chat completions."""

    user_message = "Tell me a story."
    messages = [Message(role="user", content=user_message, tool_calls=None, refusal=None)]
    model = "gemini-2.0-flash"
    response_chunks = ["Once upon a time,", " there was a cat."]

    provider = GeminiProvider()
    response = await provider.chat_completions_create(
        messages=messages, model=model, stream=True, response_chunks=response_chunks
    )

    # Consume the async generator
    chunks = []
    async def consume_stream():
        nonlocal chunks
        async for r in response:
            chunks.append(r)

    await consume_stream()

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == response_chunks[0]
    assert chunks[1].choices[0].delta.content == response_chunks[1]
    assert chunks[0].choices[0].finish_reason is None
    assert chunks[1].choices[0].finish_reason is None
    assert chunks[0].metadata["model"] == "mocked-model"
    assert chunks[1].metadata["model"] == "mocked-model"


@pytest.mark.asyncio
async def test_gemini_provider_system_message():
    """Test chat completions with a system message."""

    system_message = "You are a helpful assistant."
    user_message = "What is the capital of France?"
    messages = [
        Message(role="system", content=system_message, tool_calls=None, refusal=None),
        Message(role="user", content=user_message, tool_calls=None, refusal=None),
    ]
    model = "gemini-2.0-flash"
    response_text = "The capital of France is Paris."

    provider = GeminiProvider()
    response = await provider.chat_completions_create(
        messages=messages, model=model, response_text=response_text
    )

    assert response.choices[0].message == response_text
