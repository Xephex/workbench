# Chat streaming

`POST /v1/chat/completions` accepts `stream: true` for Server-Sent Events.
Responses contain OpenAI JSON chunks followed by exactly one `data: [DONE]`.
With `stream_options: {"include_usage": true}`, the final JSON chunk has
`choices: []` and token usage. Tool calls appear in `delta.tool_calls`.

While waiting for inference, SSE comments report `queued`, `loading`, `prompt`
or `generation` every ten seconds. These are phases, not percentage estimates.
Disconnecting cancels queued work or aborts active inference. The server waits
for native finalization before releasing the model and discards interrupted
cache state. Successful turns retain their loaded handle and cache.

Validation and model lookup errors use HTTP error responses. Inference errors
after streaming headers appear as a JSON `error` frame followed by `[DONE]`;
the HTTP status remains 200. Exception details are kept out of the response.

On slow CPUs, dcode's roughly 11k-token system prompt can take minutes to
process. Raise `LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S` above the expected
prefill time (its default is 120 seconds). SSE comments keep the transport
active but may not reset an agent framework's timer between JSON chunks.
