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

### glm: render GLM 5.3 low reasoning effort as Low

GLM 5.3 (Flash and Full) chat templates accept `low` as a reasoning
effort, but ds4 rendered a requested `low` as `Reasoning Effort: High`
for every GLM model. The official 5.3 templates select the effective
effort from `['low', 'high']` (else `max`); GLM 5.2 only branches on
`== 'high'` and has no low level.

An explicit `low` (and `minimal`, which already parses as low) now renders
`Reasoning Effort: Low` when the loaded model's template supports it. The
check runs once at model load on the `effective_reasoning_effort`
statement of the embedded `tokenizer.chat_template`, covering GLM 5.3
Flash (`glm5-next`) and Full GLM 5.3 (loading through the `glm-dsa`
shape). A missing or unrecognized template keeps the old mapping, so GLM
5.2 is unchanged. `medium`, `high`, `xhigh`, `max`, the unset default and
disabled thinking are unchanged, as is every request form that names an
effort (OpenAI `reasoning_effort`, `chat_template_kwargs`, Anthropic
`output_config`/`thinking`, Responses `reasoning`).
Verified by `test_glm_template_low_effort_detection` (template detection
on the opening lines of the official 5.3 Flash, Full 5.3 and 5.2
templates) and `test_glm_reasoning_effort_levels` (per-level mapping for
both generations). (Implements antirez/ds4#1160.)

### server: advertise loaded model modalities on /v1/models

`/v1/models` and `/v1/models/{id}` now carry an `architecture` object
with `input_modalities` and `output_modalities`. Input includes `image`
only when `ds4_engine_has_vision()` reports that the loaded engine is
ready for vision; otherwise it contains only `text`. Output remains
`text`, and compatibility aliases report the same loaded-engine
capabilities, so discovery clients can configure image support
automatically from whichever model id they resolved.
Verified by `test_model_metadata_advertises_modalities` (text-only and
vision metadata JSON shapes). (Implements antirez/ds4#1165.)

### server: reasoning deltas stream on token boundaries

Streaming `/v1/chat/completions` (and the `/v1/responses` and
`/v1/messages` reasoning surfaces) no longer split `reasoning_content`
deltas at arbitrary byte offsets mid-token. The old logic held back a
flat 7 bytes against the ` Watson` close marker, so the emit window
moved with every appended token and cut wherever it landed ("1.  T" →
"he use" → "r a" → ...). Reassembled text was correct, but clients that
consume thinking deltas individually — reasoning-length estimation,
per-token UIs, signature checks — saw garbage units, and every byte-sized
delta carried full SSE+JSON overhead.

The generator appends whole decoded pieces, so the hold now releases
whole pieces and holds back only from the piece boundary at or before a
possible close-marker prefix: `think_close_suffix_len()` finds the
longest tail that can still grow into the marker, and the cut is aligned
back to the last piece start (`think_piece_start` on all three stream
states). Cuts never land mid-piece; each delta is one or more whole
decoded tokens, mirroring the content path. The UTF-8 guard and the
close/tool marker routing are unchanged, and reassembled reasoning text
is byte-identical.

`test_openai_stream_reasoning_deltas_are_token_aligned` feeds
token-sized pieces (one ending in a byte that could start the marker)
and requires whole-piece deltas; `test_openai_chat_stream_splits_
reasoning_without_tools` now pins the token-aligned shape instead of
the old 20-byte cut. (Implements antirez/ds4#685.)

### tokenizer: bpe_emit_piece merge loop is O(n log n), not O(n^2)

The byte-level BPE merge loop rescanned every adjacent symbol pair on
every merge iteration, so a piece of n symbols cost O(n^2) rank lookups.
Real CJK prompts arrive from the pre-tokenizer as one long piece, which
made a 24k-token request spend 3-4 minutes of pure CPU inside the
tokenizer before prefill even started (39k chars measured ~90-180 s; 10x
the text cost 80x the time).

Only the two adjacencies around a merge point can change rank after a
merge, so the loop now keeps a min-heap of candidate pair ranks with lazy
validation (symbols carry a merge version; stale entries are dropped on
pop) over a doubly-linked symbol list. Pop order is (rank, position) —
lowest merge rank first, earliest pair on equal ranks — which reproduces
the original merge selection exactly, so tokenization is byte-identical.
Measured on the new `speed-bench/bpe_tokenizer_bench` (synthetic dense
merges, no model needed): 1000 chars 3.8ms -> 0.19ms, 32k chars 4.76s ->
7.3ms (~660x), with the heap and the reference loop emitting identical
token sequences at every size.

Correctness is pinned by `ds4_test --bpe`, a differential test that runs
both implementations over 2000 random pieces against a synthetic
vocabulary with colliding merge ranks (tie-breaking by position is
exercised), plus the reference O(n^2) implementation retained under test
hooks. (Implements antirez/ds4#853.)

### build: make cpu-test runs the model-free regression battery on any box

New `ds4_test_cpu` build and `make cpu-test` target: the server
parsing/rendering/streaming unit group, the GGUF type geometry check, the
BPE heap differential test, and the sampling reference comparison now run
with no GPU, no Metal, and no model, on any Linux machine (and CI). The
QA gate previously treated CPU builds as compile-only; the no-model
surface is now executable everywhere. Also adds
`make bpe-tokenizer-bench`.

### ci: GitHub Actions workflow runs the model-free battery on every push

`cpu-tests.yml` builds the CPU-only binaries with the warning-free gate
(`make cpu`), runs `make cpu-test`, the additional no-model units
(prompt prefix, web recovery, gpu args, Linux memory, layer pack), the
tokenizer benchmark, and the eval self-checks. Backend-specific suites
(Metal, CUDA, ROCm) still run on their own hardware per
QA_BEFORE_RELEASES.md; this job makes the platform-independent surface
regression-gated on every push and pull request.
