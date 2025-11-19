# Provider Streaming Usage Notes

## 总览
为了在配额系统中按 token 精确扣费，需要从 AISuite 的 Provider 输出中获取 `prompt_tokens / completion_tokens / total_tokens`。各 Provider 的流式实现状态如下：

| Provider | 是否支持实时 usage | 代码现状 | 需要的改动 |
| --- | --- | --- | --- |
| OpenAI | Responses API / Chat Completions 的非流式和流式事件都支持 `usage`（例如 `response.usage` 或 `chunk.usage`，需要 `stream_options.include_usage=True`）。 | `openai_provider.py` 已在非流式与流式路径统一把 usage 写入 `ChatCompletionResponse.metadata["usage"]`，流式仅最后一帧带 usage。 | 已实现，无需改动。 |
| DeepSeek | API 兼容 OpenAI，流式 chunk 同样可以返回 `usage`。 | `deepseek_provider.py` 已在非流式与流式路径把 usage 写入 `metadata["usage"]`，流式仅最后一帧包含 usage。 | 已实现，无需改动。 |
| DeepSeekAli | API 兼容 OpenAI（阿里云版 DeepSeek），流式 chunk 同样可以返回 `usage`。 | `deepseekali_provider.py` 已在非流式与流式路径把 usage 写入 `metadata["usage"]`，流式仅最后一帧包含 usage。 | 已实现，无需改动。 |
| CloseAI | OpenAI 协议兼容，流式 chunk 同样可以返回 `usage`。 | `closeai_provider.py` 已在非流式与流式路径把 usage 写入 `metadata["usage"]`，流式仅最后一帧包含 usage。 | 已实现，无需改动。 |
| Vercel AI | 基于 OpenAI 风格，支持 `usage`（字段可能是 `input_tokens/output_tokens` 或 `prompt_tokens/completion_tokens`）。 | `vercel_provider.py` 已通过 `_normalize_usage` 统一 usage，并在非流式与流式路径写入 `metadata["usage"]`，流式仅最后一帧包含 usage。 | 已实现，无需改动。 |
| Gemini | Gemini API 在非流式响应的 `usage_metadata` 以及流式事件的 `usageMetadata` 中提供 token 统计（`prompt_token_count/candidates_token_count/total_token_count`）。 | `gemini_provider.py` 已实现非流式与流式 usage 归一化，统一写入 `metadata["usage"]`，流式仅最后一帧包含 usage。 | 已实现，无需改动。 |
| Anthropic (Claude) | Messages API 的非流式响应 `usage` 以及 streaming 的 `message_delta/message_stop` 事件都包含 `input_tokens/output_tokens`。 | `anthropic_provider.py` 已在非流式与流式路径通过 `_normalize_usage_obj` 写入 `metadata["usage"]`，流式只在带 `finish_reason` 的最后一帧携带 usage。 | 已实现，无需改动。 |

## 对接建议
1. **统一输出**：所有 Provider 在流式/非流式均应把 token 统计写到 `ChatCompletionResponse.metadata["usage"]`，这样 AISuite 的调用者可以统一读取 `response.usage`。
2. **流式实现步骤**：
   - 在流循环中新增 `stream_usage = None`。
   - 遇到携带 usage 的 chunk 时保存：
     ```python
     if getattr(chunk, "usage", None):
         stream_usage = {
             "prompt_tokens": chunk.usage.get("input_tokens") or chunk.usage.get("prompt_tokens"),
             "completion_tokens": chunk.usage.get("output_tokens") or chunk.usage.get("completion_tokens"),
         }
         stream_usage["total_tokens"] = stream_usage["prompt_tokens"] + stream_usage["completion_tokens"]
     ```
   - 在 `finish_reason` 已确定或 `response.completed` 事件时，yield 一个 `ChatCompletionResponse`，其 metadata 包含 `usage=stream_usage`。
3. **回退方案**：对于暂时拿不到 usage 的 Provider（如果有），需要在业务层做 token 估算或限制它们用于计费场景。

该文档用于指导下一步在 AISuite Provider 层补齐流式 usage 数据，确保上层可以安全地按 token 扣费。

