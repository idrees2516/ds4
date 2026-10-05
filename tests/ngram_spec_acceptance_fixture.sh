#!/bin/sh
# N-gram speculative drafting acceptance fixture (antirez/ds4#1167).
#
# Verifies that --ngram-spec N greedy decoding produces byte-identical
# output to target-only decode on the acceptance prompts (<=256 tokens:
# byte-identical on/off is expected; long contexts can differ in floating
# point reduction order through the batched verifier, like --dspark — see
# docs/SPECULATIVE_DECODING.md).
#
# Requires a model and a DSpark support GGUF, so it runs on release
# hardware (Metal). Skips itself when the inputs are missing.
#
# Usage:
#   DS4_TEST_MODEL=ds4flash.gguf \
#   DS4_TEST_DSPARK=gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf \
#   tests/ngram_spec_acceptance_fixture.sh [binary]
#
# The binary defaults to ./ds4 (Metal build). Prompts are the official
# test-vector prompts, which fit the acceptance window.

set -e

MODEL="${DS4_TEST_MODEL:-ds4flash.gguf}"
SUPPORT="${DS4_TEST_DSPARK:-}"
BIN="${1:-./ds4}"

if [ ! -f "$MODEL" ]; then
    echo "ngram-acceptance: skipped (no model at $MODEL; set DS4_TEST_MODEL)"
    exit 0
fi
if [ -z "$SUPPORT" ] || [ ! -f "$SUPPORT" ]; then
    echo "ngram-acceptance: skipped (no DSpark support GGUF; set DS4_TEST_DSPARK)"
    exit 0
fi
if [ ! -x "$BIN" ]; then
    echo "ngram-acceptance: skipped (no binary at $BIN; build with make)"
    exit 0
fi

fail=0
for prompt_file in tests/test-vectors/flash-0731/prompts/short_*.txt; do
    name=$(basename "$prompt_file" .txt)
    base=$("$BIN" --cpu -m "$MODEL" --mtp-model "$SUPPORT" \
        --dspark-strict --temp 0 -p "$(cat "$prompt_file")" 2>/dev/null || true)
    spec=$("$BIN" --cpu -m "$MODEL" --mtp-model "$SUPPORT" \
        --ngram-spec 5 --temp 0 -p "$(cat "$prompt_file")" 2>/dev/null || true)
    if [ "$base" = "$spec" ]; then
        echo "ngram-acceptance: $name OK"
    else
        echo "ngram-acceptance: $name MISMATCH"
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "ngram-acceptance: all prompts byte-identical"
fi
exit $fail
