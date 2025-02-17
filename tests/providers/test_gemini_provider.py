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
    # 移除IS_TESTING环境变量，这样就会使用真实的API调用
    if "IS_TESTING" in os.environ:
        del os.environ["IS_TESTING"]


@pytest.mark.asyncio
async def test_gemini_provider_non_streaming():
    """Test non-streaming chat completions with real API call."""
    user_message = "What is 1+1?"
    messages = [Message(role="user", content=user_message, tool_calls=None, refusal=None)]
    model = "gemini-pro"  # 使用实际的模型名称

    provider = GeminiProvider()
    response = await provider.chat_completions_create(
        messages=messages,
        model=model
    )
    
    assert response.choices[0].message is not None
    assert isinstance(response.choices[0].message, str)
    assert len(response.choices[0].message) > 0
    assert response.metadata["model"] == model


@pytest.mark.asyncio
async def test_gemini_provider_streaming():
    """Test streaming chat completions with real API call."""
    user_message = "Count from 1 to 3."
    messages = [Message(role="user", content=user_message, tool_calls=None, refusal=None)]
    model = "gemini-pro"

    provider = GeminiProvider()
    response = await provider.chat_completions_create(
        messages=messages,
        model=model,
        stream=True
    )

    chunks = []
    async def consume_stream():
        nonlocal chunks
        async for r in response:
            chunks.append(r)

    await consume_stream()

    assert len(chunks) > 0
    assert all(chunk.choices[0].delta.content is not None for chunk in chunks)
    assert all(isinstance(chunk.choices[0].delta.content, str) for chunk in chunks)
    assert all(chunk.metadata["model"] == model for chunk in chunks)


@pytest.mark.asyncio
async def test_gemini_provider_system_message():
    """Test chat completions with a system message using real API call."""
    system_message = "You are a helpful math tutor."
    user_message = "What is the square root of 16?"
    messages = [
        Message(role="system", content=system_message, tool_calls=None, refusal=None),
        Message(role="user", content=user_message, tool_calls=None, refusal=None),
    ]
    model = "gemini-pro"

    provider = GeminiProvider()
    response = await provider.chat_completions_create(
        messages=messages,
        model=model
    )

    assert response.choices[0].message is not None
    assert isinstance(response.choices[0].message, str)
    assert len(response.choices[0].message) > 0
    assert response.metadata["model"] == model
