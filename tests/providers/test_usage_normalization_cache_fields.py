import pytest

from aisuite.providers.deepseek_provider import DeepseekProvider
from aisuite.providers.gemini_provider import _normalize_gemini_usage
from aisuite.providers.openai_provider import OpenaiProvider


@pytest.fixture(autouse=True)
def set_api_key_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-api-key")


def test_openai_normalize_chat_completions_cached_tokens():
    provider = OpenaiProvider()
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 40},
    }
    normalized = provider._normalize_usage(usage)
    assert normalized == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cache_read_input_tokens": 40,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }


def test_openai_normalize_responses_cached_tokens():
    provider = OpenaiProvider()
    usage = {
        "input_tokens": 80,
        "output_tokens": 20,
        "total_tokens": 100,
        "input_tokens_details": {"cached_tokens": 10},
    }
    normalized = provider._normalize_usage(usage)
    assert normalized == {
        "prompt_tokens": 80,
        "completion_tokens": 20,
        "total_tokens": 100,
        "cache_read_input_tokens": 10,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }


def test_deepseek_normalize_cache_hit_miss_tokens():
    provider = DeepseekProvider()
    usage = {
        "prompt_cache_hit_tokens": 10,
        "prompt_cache_miss_tokens": 90,
        "completion_tokens": 20,
    }
    normalized = provider._normalize_usage(usage)
    assert normalized["prompt_tokens"] == 100
    assert normalized["completion_tokens"] == 20
    assert normalized["total_tokens"] == 120
    assert normalized["cache_read_input_tokens"] == 10
    assert normalized["cache_write_input_tokens"] == 0


def test_deepseek_normalize_openai_cached_tokens_fallback():
    provider = DeepseekProvider()
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 5},
    }
    normalized = provider._normalize_usage(usage)
    assert normalized["prompt_tokens"] == 100
    assert normalized["completion_tokens"] == 20
    assert normalized["total_tokens"] == 120
    assert normalized["cache_read_input_tokens"] == 5


def test_gemini_normalize_cached_content_token_count_snake_case():
    usage_metadata = {
        "prompt_token_count": 100,
        "cached_content_token_count": 60,
        "candidates_token_count": 20,
        "total_token_count": 120,
    }
    normalized = _normalize_gemini_usage(usage_metadata)
    assert normalized == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_input_tokens": 60,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }


def test_gemini_normalize_cached_content_token_count_camel_case():
    usage_metadata = {
        "promptTokenCount": 100,
        "cachedContentTokenCount": 50,
        "candidatesTokenCount": 20,
        "totalTokenCount": 120,
    }
    normalized = _normalize_gemini_usage(usage_metadata)
    assert normalized == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_input_tokens": 50,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }


def test_gemini_normalize_response_token_count_snake_case():
    usage_metadata = {
        "prompt_token_count": 100,
        "cached_content_token_count": 60,
        "response_token_count": 20,
        "total_token_count": 120,
    }
    normalized = _normalize_gemini_usage(usage_metadata)
    assert normalized == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_input_tokens": 60,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }


def test_gemini_normalize_response_token_count_camel_case():
    usage_metadata = {
        "promptTokenCount": 100,
        "cachedContentTokenCount": 50,
        "responseTokenCount": 20,
        "totalTokenCount": 120,
    }
    normalized = _normalize_gemini_usage(usage_metadata)
    assert normalized == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_input_tokens": 50,
        "cache_write_input_tokens": 0,
        "cache_write_by_ttl": {
            "ephemeral_5m_input_tokens": 0,
            "ephemeral_1h_input_tokens": 0,
        },
    }
