import pytest
import os
import asyncio
from aisuite.client import Client
from aisuite.framework.chat_completion_response import ChatCompletionResponse

# Test timeout for streaming responses (15 seconds)
STREAM_TIMEOUT = 15

# Fixture for Deepseek provider
@pytest.fixture
def deepseek_client():
    return Client(provider_configs={
        "deepseek": {"api_key": os.getenv("DEEPSEEK_API_KEY")}
    })

@pytest.fixture
def deepseekali_client():
    return Client(provider_configs={
        "deepseekali": {"api_key": os.getenv("DASHSCOPE_API_KEY")}
    })

@pytest.mark.asyncio
async def test_deepseek_streaming_response(deepseek_client):
    """Test Deepseek streaming response"""
    messages = [{"role": "user", "content": "Hello, World!"}]
    
    # Capture streamed response
    stream = await deepseek_client.chat.completions.create(
        model="deepseek:deepseek-chat",
        messages=messages,
        stream=True
    )
    
    collected_chunks = []
    async for chunk in stream:
        assert isinstance(chunk, ChatCompletionResponse)
        assert len(chunk.choices) == 1
        collected_chunks.append(chunk.choices[0].delta.content)
    
    # Verify final message
    full_response = "".join([c for c in collected_chunks if c is not None])
    assert len(full_response) > 0

@pytest.mark.asyncio
async def test_deepseekali_multi_turn(deepseekali_client):
    """Test multi-turn conversation with DeepseekAli"""
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}, 
        {"role": "user", "content": "What is the population of Paris?"}
    ]
    
    stream = await deepseekali_client.chat.completions.create(
        model="deepseekali:deepseek-v3", 
        messages=messages,
        stream=True
    )
    
    collected_chunks = []
    async for chunk in stream:
        assert isinstance(chunk, ChatCompletionResponse)
        assert len(chunk.choices) == 1
        collected_chunks.append(chunk.choices[0].delta.content)
    
    full_response = "".join([c for c in collected_chunks if c is not None])
    assert len(full_response) > 0
    assert "population" in full_response.lower()

@pytest.mark.asyncio
async def test_invalid_model_error(deepseek_client):
    """Test error handling for invalid model"""
    messages = [{"role": "user", "content": "Test invalid model"}]
    
    with pytest.raises(Exception):
        await deepseek_client.chat.completions.create(
            model="invalid-model",
            messages=messages,
            stream=True
        )

@pytest.mark.asyncio
async def test_stream_timeout(deepseek_client):
    """Test streaming timeout"""
    messages = [{"role": "user", "content": "Timeout test"}]
    
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            deepseek_client.chat.completions.create(
            model="deepseek:deepseek-chat",
                messages=messages,
                stream=True
            ),
            timeout=0.1
        )
