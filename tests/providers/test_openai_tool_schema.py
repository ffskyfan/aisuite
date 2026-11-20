import json
from copy import deepcopy

import httpx
import pytest

from aisuite.providers.openai_provider import OpenaiProvider
from aisuite.providers.openai_tool_schema import project_function


def exclusive_tool():
    return {
        "type": "function",
        "name": "select",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "path": {"type": "string"},
                "id": {"type": "integer"},
                "label": {"type": "string"},
            },
            "oneOf": [{"required": ["path"]}, {"required": ["id"]}],
        },
    }


def test_closed_exclusive_fields_and_null_placeholders():
    original = exclusive_tool()
    before = deepcopy(original)
    wire, plan = project_function(original)
    assert original == before
    assert wire["strict"] is True
    root = wire["parameters"]
    assert root["required"] == ["request"]
    branches = root["properties"]["request"]["anyOf"]
    assert [branch["required"] for branch in branches] == [
        ["path", "label"],
        ["id", "label"],
    ]
    assert all(branch["additionalProperties"] is False for branch in branches)
    assert json.loads(plan.decode('{"request":{"path":"场景","label":null}}')) == {
        "path": "场景"
    }
    for malformed in ("{", "[]", '{"request":null}', '{"request":{},"extra":1}'):
        assert plan.decode(malformed) == malformed
    # An invalid call must still reach application validation with both values visible.
    assert json.loads(plan.decode('{"request":{"path":"x","id":1}}')) == {
        "path": "x",
        "id": 1,
    }


@pytest.mark.parametrize(
    "value_schema",
    [
        {"type": "object"},
        {"type": ["string", "null"]},
        {"oneOf": [{"type": "number"}, {"type": "integer"}]},
        {"oneOf": [True, {"type": "string"}]},
        {"$ref": "#/$defs/nonempty"},
    ],
)
def test_auto_strict_fallback_preserves_ambiguous_open_nullable_or_referenced_fields(
    value_schema,
):
    original = exclusive_tool()
    original["parameters"]["properties"]["label"] = value_schema
    if "$ref" in value_schema:
        original["parameters"]["$defs"] = {
            "nonempty": {"type": "string", "minLength": 1}
        }
    wire, plan = project_function(original)
    assert wire["strict"] is False and plan is None
    assert wire["parameters"]["type"] == "object"
    if "$ref" in value_schema:
        assert wire["parameters"] == original["parameters"]
    else:
        assert all(
            b["properties"]["label"] == value_schema
            for b in wire["parameters"]["anyOf"]
        )
    with pytest.raises(ValueError, match="strict function 'select'"):
        project_function({**original, "strict": True})


def test_literal_one_of_is_not_a_schema_and_explicit_nonstrict_is_respected():
    tool = {
        "name": "echo",
        "parameters": {
            "type": "object",
            "properties": {
                "oneOf": {"type": "string", "examples": [{"oneOf": [1]}]},
            },
        },
    }
    assert project_function(tool) == (tool, None)
    wire, plan = project_function({**exclusive_tool(), "strict": False})
    assert wire["strict"] is False and plan is None
    assert "anyOf" in wire["parameters"] and "request" not in wire["parameters"]


def test_nested_calls_restore_optional_fields_and_keep_required_nulls():
    tool = exclusive_tool()
    tool["parameters"]["properties"]["rows"] = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["value"],
            "properties": {
                "value": {"oneOf": [{"type": "null"}, {"type": "string"}]},
                "note": {"type": "string"},
            },
        },
    }
    wire, plan = project_function(tool)
    assert wire["strict"] is True
    rows = [{"value": None, "note": None}, {"value": "x", "note": "keep"}]
    encoded = {"request": {"path": "x", "label": None, "rows": rows}}
    assert json.loads(plan.decode(json.dumps(encoded))) == {
        "path": "x",
        "rows": [{"value": None}, {"value": "x", "note": "keep"}],
    }


def test_unwrapped_historical_calls_can_still_be_replayed():
    provider = object.__new__(OpenaiProvider)
    arguments = '{"path":"before-update"}'
    history = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "old",
                    "type": "function",
                    "function": {"name": "select", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old", "content": "ok"},
    ]
    items = provider.build_replay_view(
        "gpt-6.2-sol", history, tools=[exclusive_tool()]
    ).request_view
    assert items[0]["arguments"] == arguments
    assert items[1]["type"] == "function_call_output" and items[1]["call_id"] == "old"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_sdk_sends_projected_tools_and_replays_native_arguments(stream):
    sent = []
    arguments = '{"request":{"id":3,"label":null}}'
    calls = [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "select",
            "arguments": arguments,
        },
        {
            "type": "function_call",
            "id": "fc_2",
            "call_id": "call_2",
            "name": "echo",
            "arguments": '{"label":null}',
        },
    ]
    response = {
        "id": "resp_1",
        "object": "response",
        "model": "gpt-6.2-sol",
        "status": "completed",
        "output": calls,
    }

    def endpoint(request):
        body = json.loads(request.content)
        sent.append(body)
        if len(sent) > 1:
            return httpx.Response(200, json={**response, "output": []})
        if not stream:
            return httpx.Response(200, json=response)
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**calls[0], "arguments": ""},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 0,
                "item_id": "fc_1",
                "delta": arguments,
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 0,
                "item_id": "fc_1",
                "arguments": arguments,
            },
            {"type": "response.output_item.done", "output_index": 0, "item": calls[0]},
            {"type": "response.completed", "response": response},
        ]
        data = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=data
        )

    raw = exclusive_tool()
    tool = {
        "type": "function",
        "function": {k: v for k, v in raw.items() if k != "type"},
    }
    echo = {
        "type": "function",
        "name": "echo",
        "strict": False,
        "parameters": {"type": "object"},
    }
    messages = [{"role": "user", "content": "select"}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
        provider = OpenaiProvider(
            api_key="test", base_url="https://openai.invalid/v1", http_client=client
        )
        first = await provider.chat_completions_create(
            model="gpt-6.2-sol", messages=messages, tools=[tool, echo], stream=stream
        )
        if stream:
            # A second request can use the same tool name with a different contract.
            other = {**echo, "name": "select"}
            await provider.chat_completions_create(
                model="gpt-6.2-sol", messages=messages, tools=[other]
            )
            chunks = [chunk async for chunk in first]
            emitted = [
                call
                for chunk in chunks
                for call in chunk.choices[0].delta.tool_calls or []
            ]
            assert [call.id for call in emitted] == ["call_1", "call_2"]
            assert json.loads(emitted[0].function.arguments) == {"id": 3}
            message = chunks[-1].metadata["canonical_message"]
        else:
            message = first.choices[0].message
        assert [json.loads(call.function.arguments) for call in message.tool_calls] == [
            {"id": 3},
            {"label": None},
        ]
        assert message.reasoning_content.raw_data["output"][0]["arguments"] == arguments
        followup = [
            *messages,
            message,
            *[
                {"role": "tool", "tool_call_id": call["call_id"], "content": "ok"}
                for call in calls
            ],
        ]
        await provider.chat_completions_create(
            model="gpt-6.2-sol", messages=followup, tools=[tool, echo]
        )
    assert sent[0]["tools"][0]["strict"] is True
    assert sent[0]["tools"][0]["parameters"]["required"] == ["request"]
    assert sent[0]["tools"][1] == echo
    replay = sent[-1]["input"]
    assert [
        item["arguments"] for item in replay if item.get("type") == "function_call"
    ] == [arguments, calls[1]["arguments"]]
    assert [
        item["call_id"] for item in replay if item.get("type") == "function_call_output"
    ] == ["call_1", "call_2"]
