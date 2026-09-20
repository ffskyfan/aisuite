from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import Response

from aisuite.providers.openai_provider import OpenaiProvider


@pytest.mark.asyncio
@pytest.mark.parametrize('status,finish', [('completed', 'stop'), ('incomplete', 'length'), ('failed', 'error')])
async def test_terminal_stream_preserves_replay_usage_status_and_no_duplicate_text(status, finish):
    response = Response.model_construct(
        id='resp_1', status=status, model='gpt-5.6-terra',
        output=[
            {'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque'},
            {'type': 'message', 'id': 'msg_1', 'role': 'assistant', 'status': 'completed',
             'content': [{'type': 'output_text', 'text': 'Answer', 'annotations': []}]},
            {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1', 'name': 'read', 'arguments': '{}'},
        ],
        usage={'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30},
        incomplete_details={'reason': 'max_output_tokens'} if status == 'incomplete' else None,
    )
    # Validate nested types as real SDK responses would.
    from openai._models import construct_type
    response = construct_type(type_=Response, value=response.model_dump())
    class Stream:
        closed = False
        async def __aiter__(self):
            if status == 'completed':
                yield SimpleNamespace(type='response.output_item.done', item=response.output[1])
            yield SimpleNamespace(type='response.' + status, response=response)
        async def close(self):
            self.closed = True
    raw_stream = Stream()
    provider = OpenaiProvider.__new__(OpenaiProvider)
    provider.client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=raw_stream)))
    stream = await provider._responses_create('gpt-5.6-terra', [], stream=True,
        _replay_request_view=[], _replay_mode='responses_output')
    chunks = [chunk async for chunk in stream]
    terminal = chunks[-1]
    assert terminal.choices[0].finish_reason == finish
    assert terminal.usage['prompt_tokens'] == 10
    canonical = terminal.metadata['canonical_message']
    assert canonical.tool_calls[0].id == 'call_1'
    assert canonical.reasoning_content.raw_data['output'][0]['encrypted_content'] == 'opaque'
    text = ''.join(choice.delta.content or '' for chunk in chunks for choice in chunk.choices)
    assert text == ('Answer' if status == 'completed' else '')
    assert raw_stream.closed
    kwargs = provider.client.responses.create.await_args.kwargs
    assert kwargs['store'] is False and 'reasoning.encrypted_content' in kwargs['include']
