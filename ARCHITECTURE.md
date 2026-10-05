# DwarfStar Architecture

[README](README.md) | [CHANGELOG](CHANGELOG.md) | [QA](QA_BEFORE_RELEASES.md)

This document explains how the system fits together: the execution path a
request takes, how the backends divide the work, where the caches live,
and which invariants each subsystem must preserve. It is written for
contributors who want a map before diving into the sources; the code
itself is deliberately vertical, and per-file notes extend this map.

## Design philosophy

DwarfStar is a **narrow, whole-model inference engine**, not a general
GGUF runner. It specializes in a few model families — DeepSeek V4/V4.1
Flash (+ vision experimental), GLM 5.2/5.3 (Flash and Full), and Qwen3.8
Flash Next — and compiles their exact tensor layouts, chat templates,
quantization mixes, and kernel schedules into first-class code paths.
That specialization is the performance strategy: no dispatch-by-string at
decode time, no generic tensor plumbing, no guessing the graph. What is
lost in generality is regained in memory layout, kernel fusion, and the
freedom to schedule whole-model execution as one GPU graph.

Core principles, in priority order:

1. **Correctness before speed.** A faster path ships only with an
   explanation of its numerics; unexplained logits or KV drift is a bug.
2. **One production path per backend.** Diagnostic switches validate the
   release path; no permanent semantic variants behind flags.
3. **Small, sharp C.** Objective-C only where Metal requires it; no C++
   in the engine core.
4. **Integration-tested truth.** Model loading, prompt rendering, tool
   calls, KV state, the HTTP server, and the agent are built and tested
   together (`make test`), because that is how users run them.

## Repository layout

| Path | Role |
| --- | --- |
| `ds4.c` | The engine: GGUF loading (mmap), tokenizer, CPU reference kernels, whole-model Metal graph scheduling, sessions, KV checkpoint payloads, speculative decoding, model-shape validation. |
| `ds4_metal.m`, `metal/*.metal` | Objective-C Metal runtime + compute kernels (primary backend). |
| `ds4_cuda.cu`, `cuda/mmq/` | CUDA backend + vendored llama.cpp MMQ prefill tier (see `cuda/mmq/VENDOR.md`). |
| `ds4_rocm.cu`, `rocm/*.cuh` | ROCm (HIP) backend for Strix Halo / gfx1151. |
| `ds4_cli.c` | `ds4` frontend: REPL, one-shot prompts, distributed coordinator. |
| `ds4_server.c` | `ds4-server`: OpenAI / Responses / Anthropic / completions HTTP APIs, worker queue, streaming, tool-call mapping, disk-KV policy. |
| `ds4_agent.c` | `ds4-agent`: terminal coding agent, live tool state machine, session save/restore. |
| `ds4_bench.c`, `ds4_eval.c` | Benchmarking and the built-in eval harness. |
| `ds4_tp.c`, `ds4_distributed.c` | Tensor parallelism (multi-GPU / multi-Mac) and pipeline-parallel distributed inference. |
| `ds4_ssd.c`, `ds4_kvstore.c` | SSD model streaming and the disk KV cache. |
| `ds4_layer_pack.c`, `ds4_engram.c`, `ds4_prompt_prefix.c` | Layer packing for multi-GPU placement, V4.1 Engram tables, prompt prefix handling. |
| `gguf-tools/` | Model conversion and quantization (Python + C): DeepSeek/GLM/Qwen quantizers, imatrix tooling, manifest/validators. |
| `tests/` | Unit and live integration tests, official logit test vectors. |
| `speed-bench/` | Benchmarks and recorded results. |
| `docs/` | Platform and feature guides. |

## Execution path

### 1. Model load

`ds4_engine_open()` mmaps the GGUF and parses only the header, metadata
table, and tensor directory; weights stay in the page cache until
inference touches them. `config_validate_model()` then selects the model
shape (`g_ds4_shape`) by `general.architecture` — deepseek4/deepseek41/
glm-dsa/glm5-next/qwen4exp — and fails early on any layout mismatch:
block counts, MLA dimensions, expert counts, rope, vocab. The shape
drives compile-time-shaped code through `DS4_N_LAYER`, `DS4_N_HEAD`,
`DS4_N_EXPERT` macros, so per-model constants are not runtime branches.

Model-load-time feature detection also happens here — for example the
GLM 5.3 low-effort capability is read from the embedded
`tokenizer.chat_template` (`ds4_glm_template_supports_low_effort_str()`),
and the DSpark support model is classified (`support_model_detect()`)
as legacy MTP, DSpark, or none.

Disk-only regions (V4.1 Engram tables, Qwen n-grams) are mapped behind
the resident weights and never eagerly copied.

### 2. Prompt → tokens

The chat templates for every supported model are implemented directly in
C (`render_chat_prompt_text*()`, `qwen4_chat_open()`, GLM/DeepSeek
variants), including tool-call rendering, reasoning-effort prefixes, and
thinking-mode open/close markers. Tokenization is GPT-2-style byte-level
BPE: a regex-like pre-tokenizer splits the text, then
`bpe_emit_piece()` merges symbols per piece. The merge loop keeps a
min-heap of candidate pair ranks with lazy validation over a linked
symbol list — O(n log n) per piece, byte-identical to the historical
full-rescan selection order (see `ds4_test --bpe`).

### 3. Prefill

`ds4_session_sync()` appends prompt tokens and runs the whole-model
prefill in chunks (`--prefill-chunk`, default 2048–8192 by model/backend).
On Metal the entire layer stack executes as one graph per chunk; the CUDA
backend uses the vendored MMQ prefill tier plus fused kernels; ROCm uses
the HIP graph. Indexed attention (DSA-style top-k selection per layer)
runs inside the same pass. Progress is observable through the session
progress callbacks the frontends install.

### 4. Decode

The decode loop lives in the frontends (`ds4_cli.c`, the server worker,
the agent) and drives `ds4_session_eval()` / `ds4_session_sample()`:

* **Plain path**: sample (or argmax for greedy) → eval one token →
  repeat. Eval advances the KV cache (latent MLA KV with compressed
  rows) and produces logits for the next position.
* **Speculative path**: `ds4_session_eval_speculative*()` accepts a seed
  token plus a draft; the batched verifier evaluates the block in one
  pass and commits the longest prefix whose argmax matches. Draft
  sources: the DSpark support model (markov chain over hidden state),
  the embedded MTP blocks (GLM, Qwen), or the n-gram prompt-lookup
  drafter (`--ngram-spec`). A shared scheduler backs off exponentially
  on consecutive rejections, with credit absorbing isolated misses.

KV layout is latent-cache + per-layer compressed rows; the exact shapes
are fixed per model family and validated at load. Rewind/rollback for
speculative rejection and tool-mode boundaries restores an exact
snapshot of the cache frontier (`spec_frontier`), never a recompute.

### 5. The HTTP server

`ds4_server.c` runs one inference engine behind a worker pool. Requests
arrive on four API surfaces — `/v1/chat/completions` (OpenAI),
`/v1/responses`, `/v1/messages` (Anthropic), `/v1/completions` — and are
parsed into a single internal `request` that carries the prompt, tools,
sampling, stop sequences, and live-continuation state. The generation
loop streams text, reasoning, and tool-call deltas as protocol-native
events; a DSML/GLM/Qwen-aware streaming parser hides raw model tool
markup from clients while the exact sampled bytes are kept for cache
replay.

Session slots hold live KV. Routing tries (in order): live continuation
by tool-call ids (the Responses/Anthropic contract), exact-prefix reuse
with rewind, the disk KV cache, then cold prefill. Batched-session mode
multiplexes prefills fairly between decode quanta.

### 6. The agent

`ds4-agent` is a terminal coding agent: it renders tool schemas into
the model's native syntax, runs the server-side tool state machine
locally, and keeps long sessions practical through live KV reuse plus
disk KV checkpoints (save/restore/compact). Agent requests hit the same
engine path as the server, so agent-session behavior and server behavior
cannot drift apart.

## Backends

| Backend | Target | Entry |
| --- | --- | --- |
| Metal | Apple Silicon, 96 GB+ | `make` — whole-model graph, primary performance target |
| CUDA | DGX Spark (GB10), Ada/L40S, generic | `make cuda-spark` / `make cuda-generic` |
| ROCm | Strix Halo / gfx1151 | `make strix-halo` |
| CPU | reference/debug only | `make cpu` — correctness oracle, not a performance path |

Every backend implements the same `ds4_gpu.h` surface (tensors, graphs,
fused kernels); the engine calls the abstraction, the backend owns the
launch details. CPU builds (`-DDS4_NO_GPU`) compile the whole engine
without GPU headers and serve as the compile gate and the
model-free test surface (`make cpu-test`).

## Caches

* **Live KV (slots)**: the authoritative session state, including hidden
  reasoning and exact sampled tool bytes the client never saw. Live
  continuation by tool-call id is preferred over prefix matching because
  it preserves that hidden state.
* **Disk KV (`ds4_kvstore.c`)**: checkpointed sessions on disk, keyed by
  model id + prompt prefix. Cold saves/loads are anchored at chat
  boundaries (user/assistant markers, ChatML `<|im_start|>` role
  markers) so new sessions with the same system prompt reuse them.
* **Radix prompt cache (`rax.c`)**: prefix sharing for prompt tokens.
* **SSD streaming (`ds4_ssd.c`)**: for models larger than RAM. Routed
  experts stream from disk with popularity caching; shared-expert and
  current-layer inference overlap the loads of missing experts.
* **Image embedding cache**: multimodal requests cache vision encoder
  outputs keyed by image bytes.

## Memory policy

The engine is mmap-first: resident weights are wrapped as no-copy GPU
buffers on first use; SSD streaming pins only the working set; multi-GPU
placement packs whole layers per device (`ds4_layer_pack.c`) to price KV
per card and avoid cross-device traffic in decode. The instance lock
(`ds4_acquire_instance_lock()`) prevents two huge-model processes from
destabilizing a machine.

## Concurrency model

One engine, one inference mutex, N request workers. The server
serializes GPU work but overlaps HTTP parsing, streaming, and tool I/O.
Distributed inference splits by pipeline (layers across machines,
`ds4_distributed.c`) or by tensor parallelism (attention/expert split
across GPUs or Macs, `ds4_tp.c` with RDMA/TB5 transport).

## Testing strategy

* **Model-free units** run anywhere (`make cpu-test` + CI): server
  parsing/rendering/streaming, tokenizer differential, GGUF type
  geometry, speculative scheduler and n-gram proposal units, sampling
  reference comparison. This is the regression gate for every push.
* **Model-backed tests** (`make test`, `ds4_test`) run on release
  hardware: logit vectors against the official fixtures, KV snapshot
  round trips, speculative verify depth, Metal kernel numerics.
* **Live integration** exercises the HTTP APIs and the agent end to end.
* **QA before releases** (`QA_BEFORE_RELEASES.md`) is the procedural
  gate that runs all three layers on every backend before tagging.

## Where to read next

* `AGENT.md` — working rules for agents and humans modifying this code.
* `docs/METAL.md`, `docs/CUDA_MULTI_GPU.md`, `docs/STRIX_HALO.md` —
  backend guides.
* `docs/SERVER.md`, `docs/SSD_STREAMING.md`,
  `docs/SPECULATIVE_DECODING.md`, `docs/DISTRIBUTED.md` — feature
  guides.
* `cuda/mmq/VENDOR.md` — the vendored llama.cpp MMQ provenance.
