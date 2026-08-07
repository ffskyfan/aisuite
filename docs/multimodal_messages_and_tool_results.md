# AISuite 多模态消息与工具结果设计

## 文档状态

- 状态：已接受，待实现
- 日期：2026-08-07
- 责任模块：`aisuite`
- 首要使用场景：AI 原生游戏引擎中的游戏画面观察

## 1. 背景

当前 AISuite 及其上层 Agent 链路默认把消息内容视为字符串：

- `Message.content` 是 `Optional[str]`；
- User 消息主要使用纯文本；
- Tool Result 会被 `json.dumps` 后写入 `role: tool` 消息；
- Provider Adapter 大多假设 `content` 是字符串；
- Gemini 等路径会直接使用 `Part.from_text()` 构造请求。

这使 Agent 无法在统一上下文中接收图片。即使游戏运行时已经可以捕获 PNG，图片也无法作为 User Content 或 Tool Result 交给模型理解。

本设计在 AISuite 层建立统一的多模态内容协议，使上层业务不需要理解 OpenAI、Anthropic、Gemini 等 Provider 的具体图片格式。

## 2. 决策摘要

1. AISuite 的消息内容从“字符串”扩展为“字符串或内容块数组”。
2. User Content 和 Tool Result 使用同一套文本、图片内容块。
3. Tool 产生的图片必须保留在对应 Tool Result 中，不额外伪造普通 User 消息。
4. 图片第一阶段使用 Data URL 内嵌，不建设图像上传、对象存储或图像制品系统。
5. Provider Adapter 负责把统一内容块转换为 Provider 原生协议。
6. 视觉能力由“Provider 协议能力”和“具体模型能力”共同决定。
7. 对明确不支持视觉的模型，过滤图片块并继续执行，不报错、不调用视觉兜底模型。
8. 如果过滤后消息为空，保留简短占位文本；Tool Result 必须始终保留 `tool_call_id` 配对。
9. 暂不实现 `vision_fallback_model`，但协议应允许未来加入而不修改上层消息格式。

## 3. 目标

### 3.1 功能目标

- User 消息可以同时包含文字和图片。
- Tool Result 可以同时包含文字、结构化 JSON 文本和图片。
- 同一份 AISuite 消息可被转换为 OpenAI Responses、Vercel OpenResponses、Anthropic Messages、Gemini 等原生格式。
- 现有纯文本调用和纯 JSON Tool Result 保持向后兼容。
- 非视觉模型可以继续完成任务，不因上下文中出现图片而失败。
- 工具调用和工具结果的 ID、顺序与重放语义保持不变。

### 3.2 工程目标

- 上层 Agent、编辑器和工具实现不包含 Provider 特有格式。
- Provider 不得静默把图片当普通字符串处理，也不得把 Base64 放进 JSON 文本让模型自行猜测。
- 多模态支持同时覆盖流式和非流式调用。
- Provider 转换可以通过无网络单元测试验证。
- 新 Provider 可以复用统一内容解析和能力判断逻辑。

## 4. 非目标

第一阶段不处理以下内容：

- 图像对象存储、上传 API、签名 URL、TTL 或图像制品管理；
- 跨会话长期复用图片；
- 视频、音频、PDF 等其他模态；
- 自动选择视觉兜底模型；
- 让原本不支持视觉的模型获得真正的看图能力；
- 图像生成或图像编辑输出；
- 游戏画面捕获本身。

游戏画面捕获将在本协议完成后作为上层 Tool 接入。

## 5. 统一消息协议

### 5.1 内容类型

AISuite 对外继续采用 OpenAI 风格的消息结构，`content` 支持字符串或内容块数组：

```python
from typing import Literal, NotRequired, TypedDict


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageURLValue(TypedDict):
    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ImageURLContentPart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURLValue


ContentPart = TextContentPart | ImageURLContentPart
MessageContent = str | list[ContentPart]
```

第一阶段只要求所有 Provider 共同支持以下图片来源：

```text
data:image/png;base64,...
data:image/jpeg;base64,...
data:image/webp;base64,...
```

外部 HTTP URL 可以保留在协议中，但 Provider 不支持时不得由 AISuite 自动下载。自动下载会引入认证、超时和 SSRF 等额外问题，不属于本阶段范围。

### 5.2 User Content 示例

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "请检查这个游戏画面是否存在遮挡、裁切或布局问题。",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,...",
                    "detail": "auto",
                },
            },
        ],
    }
]
```

### 5.3 Tool Result 示例

Tool Result 使用原有的 `role: tool` 和 `tool_call_id`，只是把 `content` 扩展成内容块数组：

```python
messages = [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_gameplay_visual_probe_001",
                "type": "function",
                "function": {
                    "name": "gameplay_visual_probe",
                    "arguments": "{\"scene\":\"main\",\"capture\":\"final\"}",
                },
            }
        ],
    },
    {
        "role": "tool",
        "name": "gameplay_visual_probe",
        "tool_call_id": "call_gameplay_visual_probe_001",
        "content": [
            {
                "type": "text",
                "text": (
                    "{\"success\":true,\"width\":1280,\"height\":720,"
                    "\"frame\":360,\"game_time_ms\":6000}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,...",
                    "detail": "high",
                },
            },
        ],
    },
]
```

结构化结果在第一阶段作为 `text` 内容块中的 JSON 返回。这保持 Provider 兼容，也避免为本次需求引入额外的持久化协议。

### 5.4 旧格式兼容

以下现有输入必须继续有效：

```python
{"role": "user", "content": "普通文本"}
```

```python
{
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "{\"success\":true}",
}
```

AISuite 不应强制把所有字符串升级成单元素 `text` 数组。仅在 Provider 原生协议需要内容块时进行转换。

## 6. Tool 执行结果约定

### 6.1 现有 Tool 返回值

现有 Tool 可以继续返回普通 Python 值：

```python
return {"success": True, "count": 3}
```

`Tools.execute_tool()` 对此保持现有行为，将结果序列化成 JSON 字符串。

### 6.2 多模态 Tool 返回值

AISuite 增加一个显式的 `ToolResult` 类型，用它区分“业务字典中恰好存在 content 字段”和“多模态工具结果”：

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolResult:
    content: MessageContent
    is_error: bool = False
```

多模态 Tool 示例：

```python
return ToolResult(
    content=[
        {
            "type": "text",
            "text": "{\"success\":true,\"width\":1280,\"height\":720}",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{png_base64}",
                "detail": "high",
            },
        },
    ]
)
```

`Tools.execute_tool()` 遇到 `ToolResult` 时必须原样保留其内容块，不得再次 `json.dumps()`。

上层通过 WebSocket 或其他进程边界执行工具时，应传输等价的显式 Tool Result Envelope，不能依靠字段猜测结果类型。

本项目 Backend 与 Editor 之间使用以下 JSON Envelope；Backend 在进入 AISuite 前将它映射为 `ToolResult`：

```json
{
  "type": "agent_tool_result",
  "version": 1,
  "content": [
    {"type": "text", "text": "{\"success\":true}"},
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,...",
        "detail": "high"
      }
    }
  ],
  "is_error": false
}
```

这只是进程边界上的显式序列化形式，不是图片制品层；图片仍然位于对应 Tool Call 的返回数据中。

## 7. 能力模型

### 7.1 Provider 能力与模型能力分离

同一个 Provider 下可能同时存在视觉模型和纯文本模型，因此不能只根据 Provider 名称判断。

```python
from dataclasses import dataclass
from typing import Literal


CapabilityState = Literal["supported", "unsupported", "unknown"]


@dataclass(frozen=True)
class MultimodalCapabilities:
    user_images: CapabilityState = "unknown"
    # Tool Result 图片的 Provider 格式差异较大，未适配时安全降级。
    tool_result_images: CapabilityState = "unsupported"
    supported_image_mime_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/webp",
    )
    supports_data_urls: bool = True
```

最终能力取交集：

```text
有效能力 = Provider 协议能力 ∩ 具体模型能力
```

例如：

- Provider 的原生协议支持 Tool Result 图片，但所选模型是纯文本模型，最终仍不支持；
- 模型支持视觉，但所用旧协议无法在 Tool Result 中携带图片，最终 Tool Result 图片仍不支持；
- 同一模型改用 Responses、OpenResponses 或原生 Messages 协议后，可能获得完整支持。

### 7.2 能力来源

能力来源按优先级排序：

1. 显式模型配置；
2. Provider 动态模型元数据，例如 Vercel `/v1/models` 的 `input_modalities`；
3. Provider Adapter 已知的协议限制；
4. 未知状态下的兼容性尝试。

不能仅通过模型名称包含 `vision`、`vl` 等字符串进行判断。

### 7.3 未知能力

为了尽可能支持新模型，`unknown` 不应立即等同于 `unsupported`：

1. 首次按多模态请求发送；
2. 如果 Provider 明确返回“不支持图片/内容类型”的请求错误，允许在尚未产生模型输出的情况下重试一次；
3. 重试时按非视觉模型规则过滤图片；
4. 将结果缓存为当前进程内的临时能力判断，避免同一模型反复失败；
5. 不持久化该临时判断，避免模型能力升级后长期错误降级。

该重试只能针对可识别的输入协议或模态错误，不能吞掉认证、配额、网络和一般 Provider 错误。

## 8. 非视觉模型行为

本阶段不使用 `vision_fallback_model`。对明确不支持视觉的模型，AISuite 在 Provider 调用前过滤图片内容块。

### 8.1 混合 User Content

输入：

```python
[
    {"type": "text", "text": "请检查这张图"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
]
```

降级后：

```python
"请检查这张图"
```

### 8.2 纯图片 User Content

纯图片 User Content 过滤后不得形成非法空消息。使用统一占位文本：

```text
[image omitted: selected model does not support vision]
```

### 8.3 混合 Tool Result

如果 Tool Result 同时包含 JSON 文本和图片，则只移除图片，保留 JSON 文本及 `tool_call_id`。

### 8.4 纯图片 Tool Result

纯图片 Tool Result 必须保留调用配对：

```python
{
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "[image omitted: selected model does not support vision]",
}
```

不得删除整条 Tool Result，否则会形成悬空 Tool Call，导致重放或 Provider 请求失败。

### 8.5 可观测性

过滤图片时：

- 不向最终用户显示错误；
- 不终止 Agent 任务；
- 写入 debug 级别日志；
- 记录 Provider、模型、消息角色、过滤图片数量；
- 不记录 Base64 图片内容。

## 9. Provider 转换规则

### 9.1 OpenAI Responses

User Content：

```text
text      → input_text
image_url → input_image
```

Tool Result：

```text
role: tool
    → type: function_call_output
    → call_id: tool_call_id
    → output: [input_text, input_image]
```

当前 `_build_responses_input_items()` 已经负责把 `role: tool` 转换为 `function_call_output`，应在此处增加内容块类型转换。

### 9.2 Vercel AI Gateway

Vercel 的优先路径应使用 OpenResponses/Responses 协议，以便同一协议覆盖多个上游模型提供商。

- User 图片转换为 OpenResponses 图片输入；
- Tool Result 图片转换为多模态 `function_call_output`；
- 通过模型元数据的 `input_modalities` 判断图片输入能力；
- Provider 路由和回退模型必须具有兼容的图片能力；
- 不能在回退链中把多模态请求静默路由到纯文本模型后再失败。

### 9.3 Anthropic

User Content：

```text
text → {"type": "text", "text": ...}
image_url(Data URL)
     → {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": ...,
            "data": ...
          }
        }
```

Tool Result：

```text
canonical content[]
    → tool_result.content[]
```

Anthropic HTTP 协议会把客户端 Tool Result 放在 `role: user` 的消息容器中，但其内部语义仍是 `tool_result` block。AISuite 的统一历史继续使用 `role: tool`，该差异只存在于 Anthropic Adapter 内部。

### 9.4 Gemini

User Content：

```text
text      → Part.from_text()
image_url → inline_data / Part.from_bytes()
```

Tool Result：

- 对支持多模态 Function Response 的模型，把图片作为 `functionResponse` 的关联 parts；
- 保留 Provider 原始 function name、call ID 和 thought signature；
- 不得把图片转换成普通用户消息；
- 不支持多模态 Function Response 的模型按非视觉 Tool Result 规则过滤图片。

当前 Gemini Provider 中所有直接调用 `Part.from_text(text=msg["content"])` 的路径都需要改为共享内容转换函数，覆盖流式、非流式、历史重放和 Tool Result。

### 9.5 OpenAI-compatible Provider

OpenAI-compatible 不是统一能力声明。不同端点可能只兼容文本 Chat Completions，也可能支持 User 图片或完整 Responses API。

规则：

- User Content 图片：端点和模型支持时原样传递 OpenAI 风格内容数组；
- Tool Result 图片：只有端点明确支持多模态 Tool Result 时才传递；
- 只支持 Chat Completions 文本 Tool Result 的端点按非视觉规则过滤图片；
- 不因为 Provider 使用 OpenAI SDK 就默认认为它支持视觉或多模态 Tool Result。

该规则适用于 DeepSeek、Kimi、GLM、Groq、Mistral、Cerebras、Together、Fireworks、Nebius、Ollama、Hugging Face、xAI、CloseAI 等路径，最终能力仍以具体模型和端点为准。

### 9.6 其他原生 Provider

Bedrock、Vertex、Cohere、Watsonx 等 Provider 应通过同一套 canonical content 输入，分别实现原生转换。

未实现图片转换前，其能力必须标记为 `unknown` 或 `unsupported`，不得在转换器中通过 `str(content)` 把内容数组扁平化。

## 10. 消息重放与上下文要求

### 10.1 Tool Call 配对

- 每个 Assistant Tool Call 必须有相同 ID 的 Tool Result；
- 图片过滤不得改变 Tool Result 的 ID；
- 并行工具结果必须维持原有顺序和调用归属；
- Provider 原生 ID 映射继续沿用现有 replay metadata；
- 多模态内容不得插入 Assistant Tool Call 与对应 Tool Result 之间。

### 10.2 历史保存

第一阶段允许 Data URL 随消息历史保存，以换取最小实现复杂度。实现时应避免：

- 把整个内容数组再次 JSON 编码为字符串；
- 把 Base64 写入普通 debug/error 日志；
- 在同一请求中不必要地复制多份图片字符串；
- 在 Provider 转换前修改 canonical history 原对象。

如果后续出现数据库体积、跨会话复用或大规模多帧捕获问题，再独立评估图像引用或制品层。本设计不提前引入这些能力。

### 10.3 上下文裁剪

现有文本上下文裁剪逻辑必须识别内容数组。第一阶段可以采用简单策略：

- 文本块按原有 token 估算；
- 图片按 Provider 可用的图片 token 统计计费；
- 无法估算时使用保守固定成本；
- 裁剪整张图片，不截断 Base64；
- 不允许只保留被截断的 Data URL。

## 11. 输入校验

本阶段只需要轻量校验，不建设图像管理系统：

- 接受 `image/png`、`image/jpeg`、`image/webp`；
- 校验 Data URL 格式和 Base64 可解码性；
- 限制单条消息图片数量；
- 限制单张图片解码后字节数；
- 拒绝未知或主动内容 MIME；
- 不在日志和异常文本中回显 Base64；
- `detail` 只允许 `auto`、`low`、`high`。

具体大小限制作为 AISuite Client 配置提供，避免硬编码到 Provider 转换器。

## 12. 代码改动范围

### 12.1 AISuite Framework

建议新增：

```text
aisuite/framework/content.py
```

职责：

- 内容块类型；
- Data URL 解析；
- 内容规范化；
- 图片检测与过滤；
- 占位文本生成；
- Provider 转换共享工具；
- 轻量输入校验。

修改：

```text
aisuite/framework/message.py
aisuite/framework/message_normalizer.py
aisuite/provider.py
```

### 12.2 Tool Runner

修改：

```text
aisuite/utils/tools.py
```

要求：

- 新增显式 `ToolResult` 类型；
- 普通返回值保持 JSON 序列化；
- 多模态 `ToolResult.content` 原样进入 Tool 消息；
- Tool Result 图片保持 `tool_call_id`；
- 流式和非流式 Agent Loop 行为一致。

### 12.3 Provider Adapter

第一批：

```text
aisuite/providers/openai_provider.py
aisuite/providers/vercel_provider.py
aisuite/providers/anthropic_provider.py
aisuite/providers/gemini_provider.py
```

第二批按协议族扩展：

```text
aisuite/providers/deepseek_provider.py
aisuite/providers/kimi_provider.py
aisuite/providers/glm_provider.py
aisuite/providers/aws_provider.py
aisuite/providers/google_provider.py
aisuite/providers/message_converter.py
```

其余 Provider 在共享协议完成后逐个增加能力声明与转换测试。

## 13. 实施阶段

### 阶段 A：协议基础

- 新增统一内容块与 `ToolResult` 类型；
- 扩展 `Message.content` 类型；
- 修改 `MessageNormalizer`，确保内容数组不会被字符串化；
- 实现 Data URL 解析、校验和非视觉过滤；
- 保持全部旧测试兼容。

### 阶段 B：核心 Provider

- OpenAI Responses；
- Vercel OpenResponses；
- Anthropic Messages；
- Gemini 原生；
- 覆盖流式、非流式和历史重放路径。

### 阶段 C：当前产品模型

对当前模型注册表中的模型建立契约测试：

- GPT 5.5、GPT 5.4；
- Claude Opus 4.6、Claude Sonnet 4.6；
- Gemini 3 Flash、Gemini 3.1 Pro；
- DeepSeek V4 Flash、DeepSeek V4 Pro；
- Kimi K3；
- GLM 5.2。

视觉模型验证图片传递；非视觉模型验证图片过滤后任务继续执行。

### 阶段 D：上层接入

- Backend Agent History 支持内容块；
- WebSocket Tool Result 支持内容数组；
- Editor Chat 支持图片上下文；
- 接入 `gameplay_visual_probe`；
- Native Player 截图以 Data URL 进入 Tool Result。

## 14. 测试矩阵

每个支持图片的 Provider Adapter 至少覆盖：

| 场景 | 预期结果 |
| --- | --- |
| 纯文本 User 消息 | 与现有行为一致 |
| 文本加一张 User 图片 | 转换为 Provider 原生多模态输入 |
| 纯图片 User 消息 | 视觉模型收到图片 |
| 文本 Tool Result | 与现有行为一致 |
| JSON 文本加一张 Tool Result 图片 | 图片属于对应 Tool Result |
| 纯图片 Tool Result | 图片属于对应 Tool Result |
| 两个并行 Tool Call，各返回不同图片 | call ID 与图片不串线 |
| 历史中包含多模态 Tool Result | 重放顺序和 Provider metadata 正确 |
| 流式模型请求 Tool，Tool 返回图片 | 下一轮流式响应正常 |
| 非流式模型请求 Tool，Tool 返回图片 | 下一轮非流式响应正常 |
| 非视觉模型收到混合内容 | 仅过滤图片，保留文字 |
| 非视觉模型收到纯图片 Tool Result | 保留 call ID 和占位文本 |
| 未知模型拒绝图片 | 只针对模态错误重试一次文本降级 |
| 非法 Base64 或 MIME | 调用 Provider 前失败 |
| 日志与异常 | 不包含 Base64 正文 |

Provider 转换测试必须使用 mock client 或快照测试，不依赖真实 API Key。真实 Provider 端到端测试作为可选契约测试运行。

## 15. 验收标准

实现完成后应满足：

1. AISuite Client 可直接发送包含 Data URL 图片的 User Content。
2. AISuite Tool Runner 可返回包含图片的 `ToolResult`。
3. 图片始终属于对应的 Tool Result，不产生伪造的普通 User 图片消息。
4. OpenAI/Vercel、Anthropic、Gemini 至少各有一条通过的 User 图片与 Tool Result 图片测试。
5. 当前视觉模型能够基于 Tool 返回图片生成有效描述。
6. 当前非视觉模型在同一输入下不会因图片失败，并保留文本与工具调用配对。
7. 原有纯文本、工具调用、流式、usage 和 replay 测试全部通过。
8. Provider Adapter 不会静默字符串化、丢失或错配受支持模型的图片。
9. 日志、错误和 telemetry 不泄露 Base64 图片正文。
10. 上层不需要根据 Provider 名称构造不同消息格式。

## 16. 后续演进

以下能力只有在真实需求出现后再评估：

- `vision_fallback_model`：将图片交给视觉模型并把描述作为原 Tool Result 的文本补充；
- 图像引用与制品层：解决大型图片、数据库膨胀、跨会话复用和多帧捕获；
- 图片去重和内容寻址；
- 视频、音频、PDF 内容块；
- 多帧游戏观察和视频理解；
- 根据模型价格、图片 token 和延迟自动选择 `detail`；
- 视觉模型回退链的模态一致性检查。

这些能力不得改变本设计中的核心语义：Tool 产生的媒体仍然属于对应 Tool Result。

## 17. 参考资料

- OpenAI Responses API：<https://platform.openai.com/docs/api-reference/responses>
- Anthropic Tool Result：<https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls>
- Gemini Function Calling：<https://ai.google.dev/gemini-api/docs/function-calling>
- Vercel OpenResponses：<https://vercel.com/docs/ai-gateway/sdks-and-apis/openresponses>
- Vercel Image Input：<https://vercel.com/docs/ai-gateway/sdks-and-apis/openresponses/image-input>
- Vercel Models and Providers：<https://vercel.com/docs/ai-gateway/models-and-providers>
