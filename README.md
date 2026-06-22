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
model = "default"

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
LLM7_MODEL            Default model. Defaults to gpt-5.5.
PROXY_HOST            Local bind host. Defaults to 127.0.0.1.
PROXY_PORT            Local bind port. Defaults to 5011.
AGENTIC_TOOL_PROMPT   Set to 0 to disable the extra tool-awareness system hint.
```
