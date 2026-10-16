# Local server

Run `python workbench.py setup --yes`, then `python workbench.py serve`.
On Windows use `py -3.14`. The server binds to `127.0.0.1:8080` and loads a
model on its first request. Only Windows and Linux are supported here.

Commands use the same HTTP API:

    python workbench.py chat --model qwen3.5-4b
    python workbench.py ask "Explain a binary search" --json
    python workbench.py embed "a short document" "another document" --json
    python workbench.py rank "bicycles" "a car" "a bicycle" --top-n 1 --json
    python workbench.py status --json

Add `--server http://127.0.0.1:8080` to use an explicit server. Inference
commands reuse a running server; otherwise they start a temporary loopback
server and stop only that process on exit. Status never starts a server.
Chat retains history for the session and saves it under `data/chats`.
`/reset` starts a fresh conversation. `--think` enables reasoning; it is off
by default on CPU-tier machines. `--json` emits complete JSON (chat emits one
JSON object per reply); progress goes to stderr in normal text mode.

A cold long agent prompt can take many minutes on CPU. The API sends SSE
comments for queued/loading/prompt/generation states every ten seconds.
"Reading the prompt" is activity feedback, not a percentage. The resident
model keeps KV state and a bounded RAM prefix cache between turns. LRU
unloading or cancellation drops that cache. A framework that times only data
chunks may still need a longer timeout: dcode uses
`LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S`. Prefer a GPU or a shorter system
prompt for dcode/Forge; the recorded 11.4k-token CPU prompt took 26 minutes.
Disconnecting cancels queued work or aborts active inference; the server waits
for native finalization before releasing the model.

`serve --share` binds all interfaces and generates/reuses
`.workbench/api-key.txt`; `WORKBENCH_API_KEY` overrides the file. Clients send
`Authorization: Bearer KEY`. Every route requires the key when one is set.
Any non-loopback `--host` without a key is refused. Keys are never printed.
Loopback binding remains the default; sharing does not enable downloads.

API routes:

- GET `/v1/models`: installed, runnable model IDs and capabilities.
- POST `/v1/chat/completions`: messages, tools/tool_choice, stream,
  stream_options.include_usage, max_tokens, sampling settings, think,
  response_format (text, json_object or json_schema).
- POST `/v1/embeddings`: model, input (string or list), encoding_format
  (float or base64). Token arrays and dimension reduction are not supported.
- POST `/v1/rerank`: model, query, documents (strings or objects with text),
  optional top_n and return_documents. Results retain original indexes and
  sort by descending relevance_score; ties retain input order.
- GET `/wb/status`: loaded models, memory reservations, device verification,
  FIFO queue depth, physical core count and optional trace viewer URL.
- GET `/wb/models`: all catalog entries and installation/runnable state.
- POST `/wb/models`: action install/remove plus model, or cancel plus job_id.
- GET `/wb/jobs`: durable progress, terminal results and interrupted jobs.

Messages accept strings or lists of text parts for every role. Assistant
tool-call history may have null content; tool replies require tool_call_id.
Vision models accept user image_url parts containing PNG/JPEG/WebP base64
data URLs; the catalog mmproj is loaded with the model. Remote URLs, local
file URLs and audio parts are rejected. JSON schema uses native grammar.
Tools and non-text response_format cannot be combined. Auto choice uses the
model template and parser; none disables tools. Required or a named choice
uses a grammar-constrained single-call envelope, held until complete before
emitting tool_calls. Tools are returned,
never executed by the server. A streamed error is an error data frame then
[DONE]; clients must inspect it even when HTTP status is 200.

RAM reserves 4 GB for the system and VRAM reserves 0.8 GB. Cache memory is
charged separately. Memory numbers are catalog estimates until matching
measurements are supplied by tune. Small embedding/reranker models use GPU
space when available alongside chat, otherwise CPU. Unknown native device
information is reported as unverified, never as proven acceleration.

Model changes are jobs. Closing a browser leaves jobs running. Cancelling a
fetch takes effect at the next chunk boundary and preserves resume state.
A server restart marks unfinished jobs interrupted; repeat the install to
resume. A model with active or queued requests cannot be removed. The models
command delegates to the server when it is running. Stop the server before
setup replaces packages.

Every request writes OTel spans to `data/traces/spans.jsonl`. If the tracing
pack is installed, the server starts a local Phoenix viewer and exports new
spans to it. Its URL appears in status. Phoenix telemetry/external resources
are disabled, and the viewer binds only to loopback. Startup failure keeps
file tracing available; inspect `data/phoenix/server.log`. No historical
file replay occurs automatically. `serve --no-trace-viewer` skips the viewer.

Inputs/outputs are recorded by default. `serve --no-trace-content` disables
content for all requests. `X-Workbench-Trace-Content: false` disables it for
one request/tab. This does not disable chat transcript saving by the chat
command. Timing, model, sampled token counts, memory and error classes remain.
No exception message or authorization header is recorded in spans.

Langfuse is optional and off by default. Put an enabled langfuse object with
host, public_key and secret_key in `data/trace.json` to add OTLP/HTTP export to
`<host>/api/public/otel/v1/traces`. Keep this user-data file private. This is an
explicit external network opt-in; local traces and Phoenix remain enabled.

Developer checks: `python3.14 tools/test_server.py`. The runner activates the
pack before importing dependencies. It is a file so Windows Phoenix and
strawberry source inspection works. Unit tests use a fake engine. Set
WORKBENCH_TEST_MODEL to an existing small chat GGUF to enable the single
optional real-model test; it checks two API turns and reduced prompt work on
the second turn and never downloads a model.
