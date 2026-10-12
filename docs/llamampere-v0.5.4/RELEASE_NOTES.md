# llamAmpere v0.5.4

A crash-fix release on top of v0.5.3. There are no new features and no new flags, and no speed change.

- **The server no longer stops with `failed to remove sequence N` when it serves several requests at once.** With
  `--parallel 3` or more and the MTP drafter on (the default), a Qwen3.8 model (Swift, ATX) could abort with
  `failed to remove sequence N with p0=...` when several requests started together. It happened when the server moved
  a waiting request's saved state to another memory slot: the move kept only the latest state and dropped the earlier
  steps the drafter needs to undo a rejected guess, so the next rejected guess for that request failed. The move now
  keeps those earlier steps. The prompt cache was never involved, so `--cache-ram 0` was not a workaround.
  - Swift 1.5 IQ4_XS-M, tq5_0 K / turbo4 V, 8,192 tokens per slot, several requests starting at once and sharing
    prompt prefixes, 5,120 tokens generated per request at temperature 1.0: v0.5.3 aborted in 4 of 4 runs that hit the
    move (`--parallel 4` and `--parallel 8`). v0.5.4 finished all 32 requests across the same 4 runs, with no abort.
    Peak VRAM 22,850 MiB at `--parallel 8`.
  - A small Qwen3.5 0.8B MTP model with `--parallel 8` and 10 clients starting at once: v0.5.3 aborted in 6 of 6
    runs, v0.5.4 completed 15 of 15.
  - A new check in `tests/test-recurrent-state-rollback.cpp` (gate H) moves a waiting request's state and checks that
    undoing a step gives exactly the same result as a fresh run. v0.5.3 fails it on the Qwen3.8 layouts; v0.5.4 passes.

What stays the same:
- With `--parallel 1` the changed code does not run. At temperature 0, Swift 1.5 with tq5_0 K / turbo4 V writes the same
  5,000 tokens as v0.5.3, byte for byte.
- Decode speed is unchanged. At `--parallel 8`, runs in the order v0.5.3, v0.5.4, v0.5.3 gave 19.42, 19.32 and 19.36
  tok/s per request (144.2, 143.4 and 143.7 tok/s in total).
- No GPU kernels changed.

The v0.5.3 known issue about `--parallel 8` aborting at startup is fixed by this release.
