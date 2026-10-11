# llamAmpere v0.5.2

A point release on top of v0.5.1. Several prefill and SJ-KVaRN fixes were added, improving reliability and speed.

- The RAM prompt cache keeps the newest 4 context checkpoints of each entry plus one per doubling of the distance from
  the end. `--cache-ram-checkpoints 0` keeps all of them (the v0.5.1 behaviour).
- `--parallel 1` decode speed at 100K depth is unchanged from v0.5.1.

## Optional: SJ-KVaRN transform files

SJ-KVaRN can use a per-model transform file in place of its built-in rotation (`--sjkvarn-transform FILE|URL`). Each
file is calibrated for one model; a URL is downloaded to the cache on first start. It is off unless you pass it.

Measured KL divergence vs a q8_0/q8_0 KV cache, with the transform divided by without it (held-out agentic
trajectory, 95% CI; below 1 = closer to q8_0):

| Model | Transform file | 3/2t | 3/3t | 4/4 |
|---|---|---|---|---|
| ATX-4-XS (`jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF`) | `Qwen3.8-27B-ATX-4-XS.sjwu` | **0.87** [0.81, 0.92] | 0.96 [0.86, 1.09] | 0.94 [0.87, 1.03] |
| Swift 1.5 IQ4_XS-M (`jakeatx/ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M-GGUF`) | `ATX-Swift-1.5-Qwen3.8-27B-Uncensored-IQ4_XS-M.sjwu` | 1.01 [0.91, 1.11] | 0.93 [0.83, 1.04] | 1.07 [0.96, 1.21] |
SA2_ROW
Only ATX-4-XS at 3/2t shows a clear gain (13% lower KL). Everywhere else the difference is within noise, so the
transform stays off by default. On ATX-4-XS it costs about 2.5% decode speed at 100K depth.

Example (ATX-4-XS, SJ-KVaRN 3/2t):

```bash
./build-sm86/bin/llama-server \
  --hf-repo jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF \
  --hf-file Qwen3.8-27B-ATX-4-XS.gguf -c 262144 \
  -ngl 99 -fa on -ctk sjkvarn3 -ctv sjkvarn2 --sjkvarn-body-type auto -b 4096 -ub 1024 --parallel 1 \
  --sjkvarn-transform https://huggingface.co/jakeatx/Qwen3.8-27B-ATX-IQ4_XS-M-GGUF/resolve/main/Qwen3.8-27B-ATX-4-XS.sjwu
```

Prompt caching and slot save/restore work with a transform. A saved state records which transform it was made with
and is not restored under a different one. The file adds about 64 MiB of VRAM.
