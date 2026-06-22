# LLM7 Codex Python Proxy

A small Python proxy that lets the Codex app talk to LLM7 through an OpenAI-compatible local endpoint.

It uses the OpenAI Python SDK with `base_url` pointed at LLM7:

```txt
Codex app -> /v1/responses -> local proxy -> OpenAI SDK -> LLM7
```

The proxy translates Codex Responses API requests into LLM7 chat completions, preserves tool definitions, streams function-call argument events back to Codex, and adds a small agentic tool hint so the model knows project/tool access happens through the app.

## Setup

Requires Python 3.10+.

1. Install requirements:

```bat
python -m pip install -r requirements.txt
```

2. Add this to your Codex `config.toml`:

```toml
model_provider = "llm7proxy"
model = "gpt-5.5"

[model_providers.llm7proxy]
name = "LLM7 Python Proxy"
base_url = "http://127.0.0.1:5011/v1"
env_key = "LLM7_API_KEY"
wire_api = "responses"
request_max_retries = 2
stream_max_retries = 2
stream_idle_timeout_ms = 300000
```

On Windows, the config is usually:

```txt
C:\Users\YOUR_USERNAME\.codex\config.toml
```

The same config block is also saved in `python-codex-llm7.config.toml`.

3. Start the proxy and keep this terminal open:

```bat
set LLM7_API_KEY=unused
python llm7_codex_proxy.py
```

Use a real LLM7 key for better limits:

```bat
set LLM7_API_KEY=your_real_key_here
python llm7_codex_proxy.py
```

Get a key from:

```txt
https://dash.llm7.io/
```

4. Launch or restart the Codex app.

You are ready to use Codex through the LLM7 proxy.

## Endpoints

```txt
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/responses
```

The local server runs at:

```txt
http://127.0.0.1:5011
```

## Options

```txt
LLM7_API_KEY          LLM7 token. Defaults to unused.
LLM7_BASE_URL         LLM7 base URL. Defaults to https://api.llm7.io/v1.
LLM7_MODEL            Upstream LLM7 model for GPT aliases. Defaults to default.
LLM7_MODEL_ALIASES    Extra comma-separated model names to show in /v1/models.
PROXY_HOST            Local bind host. Defaults to 127.0.0.1.
PROXY_PORT            Local bind port. Defaults to 5011.
AGENTIC_TOOL_PROMPT   Set to 0 to disable the extra tool-awareness system hint.
AGENTIC_TOOL_PROMPT_MAX_TOOLS
                      Defaults to 20. Limits tool names shown in the extra hint.
AGENTIC_TOOL_PROMPT_DESCRIPTIONS
                      Defaults to 0. Set to 1 to include short tool descriptions in the hint.
LLM7_SAFE_MODE        Defaults to 1. Sends only LLM7-safe chat parameters upstream.
LLM7_EXTRA_BODY_PASSTHROUGH
                      Defaults to 0. Set to 1 only if your upstream supports extra OpenAI fields.
LLM7_TEXT_TOOL_FALLBACK
                      Defaults to 1. Converts JSON text tool calls into real Codex function calls.
LLM7_FORCE_COMMAND_FALLBACK
                      Defaults to 1. If the model promises action but emits no tool call, starts with a safe command-tool inspection.
CODEX_PROXY_DEBUG     Defaults to 1. Set to 0 to disable sanitized incoming/upstream JSON dumps.
CODEX_PROXY_DEBUG_DIR Defaults to debug-dumps.
```

Codex can use `model = "gpt-5.5"`, but the proxy sends GPT-style model aliases upstream as the LLM7 model in `LLM7_MODEL`. By default that upstream model is `default`, because many LLM7-compatible endpoints reject raw GPT model IDs with a 400.

The proxy advertises common GPT/O/Codex dropdown aliases from `/v1/models`. Any alias that is not one of LLM7's native `default`, `fast`, or `pro` models is mapped upstream to `LLM7_MODEL`.

## Debug Codex Requests

The proxy writes sanitized JSON captures to `debug-dumps/` by default. To disable dumps, run:

```bat
set CODEX_PROXY_DEBUG=0
set LLM7_API_KEY=unused
python llm7_codex_proxy.py
```

Do not publish these dumps because prompts and project context may still be present.
