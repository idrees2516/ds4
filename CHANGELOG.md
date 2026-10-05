# Changelog

All notable changes to DwarfStar (ds4) in this repository are documented here,
one entry per change, newest first. Every entry states the user-visible
behavior, the components touched, and how the change is verified.

The format follows the repository's own conventions (see CONTRIBUTING.md):
prose summaries tied to the exact behavior change, plus the tests that pin it.

## 2026-10 (production-ready hardening series)

### server: report the Anthropic `stop_sequence` when a client stop matches

`/v1/messages` now reports `stop_reason: "stop_sequence"` with the matched
string in `stop_sequence` when a client `stop_sequences` entry ends the
output, in both the full response and the streamed `message_delta`.
The matched sequence is recorded on the request when the decode loop finds
it (`request.matched_stop`), freed with the request, and reported through
`anthropic_stop_reason()` / `anthropic_final_response()` /
`anthropic_sse_stop_live()`. Other stop reasons (`end_turn`, `max_tokens`,
`tool_use`) and the OpenAI-style `finish_reason` are unchanged. The
earliest-position match wins and ties keep client list order.
Verified by `test_anthropic_stop_sequence_reported` (full response, plain
`end_turn` fallback, and streamed `message_delta`), plus the extended
`test_stop_list_parses_all_sequences` / `test_stop_list_streaming_holds_and_
trims_stop_text` checks that the matcher reports which entry matched.
(Implements antirez/ds4#1161.)

### server: Anthropic clients can now see a failed turn

A turn that ends with `finish == "error"` is no longer indistinguishable
from a normal completion on the Anthropic API. Anthropic's `stop_reason`
vocabulary has no error member, so the failure is surfaced out-of-band:
the non-streaming message object carries an `"error"` field
(`{"type":"api_error","message":...}`) and the streaming path emits the
existing `sse_error_event` (`event: error`, Anthropic shape) before the
normal `message_delta` / `message_stop` terminator. `stop_reason` stays
`end_turn` so clients that ignore unknown fields keep working, while clients
that look can tell. `anthropic_final_response()` and
`anthropic_sse_finish_live()` take the generation error text; call sites
pass the worker's `err` buffer.
Verified by `test_anthropic_error_finish_is_visible` (both surfaces, plus a
clean stream termination after the error event).
(Implements the out-of-band option proposed in antirez/ds4#1164.)

### server: unterminated tool call defers to the parse-failure recovery

When the model opens a tool envelope and never closes it before the stop
token, the generator no longer sets `finish = "error"`. An unclosed
envelope is a model output shape (truncated or rambling generation), not a
server fault: leaving `finish` alone lets the existing
`recovered_tool_parse_failure` degrade path handle it exactly like an
uncloseable-but-complete envelope — raw text returned as assistant content,
`finish_reason: "stop"`, no `tool_calls`, with a WARNING log line and a
trace event so operators still see it. Genuine faults keep their error:
the enclosing guard still excludes `finish == "error"`, `finish == "length"`
and client stops, and the tool-error continuation route is tried first.
Verified by `test_unterminated_tool_call_degrades_to_assistant_text`, which
pins the parser contract (recovery applies for non-error finishes, never
overrides a genuine `error` finish).
(Implements antirez/ds4#1163.)

### server: report `max_output_tokens` as the Responses incomplete reason

`/v1/responses` now reports `incomplete_details.reason` as
`"max_output_tokens"` when output is cut by the token budget, in both the
full response and the streamed `response.incomplete` event. The previous
value `"max_tokens"` is not a member of the OpenAI Responses vocabulary, so
clients that check for `"max_output_tokens"` (Codex and other
Responses-native clients) did not see the truncation.
Verified by `test_responses_incomplete_reason_is_max_output_tokens`, which
asserts the new string is present and the old one absent on both paths.
(Implements antirez/ds4#1162.)

### gguf: fix the iq1_s and iq4_nl block sizes in the type table

The GGUF type table had two wrong entries, taken from neither ggml nor
gguf-py's `GGML_QUANT_SIZES`:
- `iq4_nl` was 256 values in 50 bytes; `block_iq4_nl` is 32 values in
  18 bytes (a half scale plus 16 bytes of nibbles).
- `iq1_s` was 110 bytes per block; `block_iq1_s` is 50 bytes (a half
  scale, 32 bytes of qs, 16 bytes of qh).

With the old entries every iq4_nl tensor was sized at about a third of
its real length (and iq1_s at more than double), so third-party GGUFs
storing n-gram tables or weights in these types failed to load or decoded
garbage. The official ds4 GGUFs use neither type; ISTA-DASLab's
Qwen3.8-Flash-Next GSQ-RCO GGUF stores its n-gram table and some weights
as iq4_nl and now decodes correctly.

Adds `ds4_test --gguf-types`, a no-model check of every quantized type's
block geometry against sizes spelled from ggml's block structs.
(Implements antirez/ds4#1156.)

### docs: fix "gaol" -> "goal" in the README hardware section

Typo fix only. (antirez/ds4#1173.)
