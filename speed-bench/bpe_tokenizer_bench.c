/* BPE tokenizer merge-loop benchmark: heap (O(n log n)) vs the historical
 * full-rescan loop (O(n^2)) on long single pre-tokenized pieces.
 *
 * Long CJK prompts arrive from the pre-tokenizer as one huge piece, which is
 * exactly the shape that made the quadratic loop take minutes on 24k-token
 * requests (antirez/ds4#853). This benchmark measures both implementations
 * on the same synthetic pieces and reports per-size timings plus the
 * speedup, so regressions in merge-loop scaling are visible without a model.
 *
 * Build: make bpe-tokenizer-bench (CPU-only, no GPU or GGUF required).
 */
#include "../ds4.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void ds4_test_bpe_vocab_build(void *vocab);
void ds4_test_bpe_vocab_free(void *vocab);
void ds4_test_bpe_emit(const void *vocab, const char *ptr, uint64_t len,
                       ds4_tokens *out, bool reference);

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint64_t token_checksum(const ds4_tokens *t) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < t->len; i++) {
        h ^= (uint64_t)(uint32_t)t->v[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(void) {
    char vocab[4096];   /* opaque ds4_vocab storage */
    ds4_test_bpe_vocab_build(vocab);

    /* Piece sizes that cross the range where the quadratic loop starts to
     * dominate real requests: 1k..32k characters. The synthetic alphabet is
     * small so merges stay dense, like real CJK text. */
    static const int sizes[] = {1000, 2000, 4000, 8000, 16000, 32000};
    printf("%10s %14s %14s %9s %12s\n",
           "chars", "heap (ms)", "rescan (ms)", "speedup", "tokens");

    uint64_t rng = 0x9e3779b97f4a7c15ULL;
    char *piece = malloc(65536);
    if (!piece) return 1;

    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        const int n = sizes[s];
        for (int i = 0; i < n; i++) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            piece[i] = (char)('a' + (int)(rng % 5));
        }
        ds4_tokens fast = {0};
        double t0 = now_sec();
        ds4_test_bpe_emit(vocab, piece, (uint64_t)n, &fast, false);
        double t_fast = now_sec() - t0;

        ds4_tokens ref = {0};
        t0 = now_sec();
        ds4_test_bpe_emit(vocab, piece, (uint64_t)n, &ref, true);
        double t_ref = now_sec() - t0;

        const bool same = fast.len == ref.len &&
            (fast.len == 0 || !memcmp(fast.v, ref.v, (size_t)fast.len * sizeof(int)));
        const uint64_t checksum = token_checksum(&fast);

        printf("%10d %14.2f %14.2f %8.1fx %12d %s (checksum=%016" PRIx64 ")\n",
               n,
               t_fast * 1000.0,
               t_ref * 1000.0,
               t_ref / (t_fast > 0 ? t_fast : 1e-9),
               fast.len,
               same ? "match" : "MISMATCH",
               checksum);

        free(fast.v);
        free(ref.v);
    }

    free(piece);
    ds4_test_bpe_vocab_free(vocab);
    return 0;
}
