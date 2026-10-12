# llamAmpere v0.5.3

A bug-fix release on top of v0.5.2. There are no new features and no new flags.

- **SJ-KVaRN prompt cache restores now match a fresh read.** In v0.5.2, a conversation resumed from the prompt cache
  (RAM, or the disk copy after a restart) could answer differently from the same conversation read fresh, because
  the restored cache had compressed a few more tokens than a fresh read had at that point. A resumed prompt now
  re-reads from exactly the state a fresh read has there. In our restore test at temperature 0 (two server starts
  sharing a disk cache), turns that differed from the fresh answer went from 6 of 24 to 0 of 24 at 4/4 and from 4 of
  24 to 0 of 24 at 3/3. Saved SJ-KVaRN states keep a few more exact tokens (about 24% more bytes per token in a short
  1,537-token unit test; the extra is fixed in size, so the share shrinks at long context).
- **The disk prompt cache works with `--mmproj`.** With a vision projector loaded, every prompt counted as an image
  prompt, so nothing was ever written to or read from the disk cache. Text-only prompts now use it; prompts with
  images still stay in RAM only (issue #35).
- **The disk cache says when it stops writing.** When the disk is too full for the cache's free-space floor, the
  server now logs it at startup and while skipping writes (at most once a minute), instead of a single warning.
  `--cache-disk-min-free` also accepts `NG` (GiB) and `N%`. What gets written is unchanged.
- **No false "draft vocabulary shortlist not applied" warning at startup** (issue #23). The MTP drafter always used
  the shortlist; the warning came from a setup step and is gone. A real fallback still warns.
- **Builds for RTX 50-series (Blackwell, sm_120).** v0.5.2 failed to compile for `-DCMAKE_CUDA_ARCHITECTURES=120`
  (one GPU file, `mmvq.cu`). Fixed and op-tested on an RTX 5070 (CUDA 13.2, 0 failures). The RTX 3090 build is
  unchanged: the GPU machine code of that file is byte-identical. RTX 40-series (`89`) is compile-checked only. 40- and
  50-series cards are not tuned for; see the build note in the README.

What stays the same and what changes in output:
- With tq5_0 K / turbo4 V (the fastest README command), answers are token-for-token identical to v0.5.2 (temperature 0,
  5,000 tokens after a 60K-token prompt and 7,000 tokens after a 2K-token prompt).
- With SJ-KVaRN, a fresh read of a long prompt is now cut into chunks at 1,024-token boundaries instead of 128-token ones
  (the first fix above needs this). That changes the rounding slightly, so at temperature 0 the answer is not
  token-for-token the same as v0.5.2's. It is no worse: over a 110K-token read, the KL divergence from a q8_0 reference
  changed by -0.000028 at 4/4 (95% interval -0.000061 to +0.000002) and -0.000002 at 3/3 (-0.000059 to +0.000057),
  both not significant. (That was measured on a broader version of this change that also moved when tokens get
  compressed; the shipped version only moves the chunk boundaries.) With that one change switched off, v0.5.3 matches v0.5.2 token for token.
- Decode speed at 100K depth is the same as v0.5.2 (see the table below).

Decode speed, RTX 3090 Ti at 350 W, Swift 1.5 IQ4_XS-M, fastest README settings (tq5_0 K / turbo4 V, MTP), prompt
about 100K tokens, 5,120 tokens generated per job at temperature 1.0, runs in the order v0.5.2, v0.5.3, v0.5.2:

| job | v0.5.2 (1st) | v0.5.3 | v0.5.2 (2nd) |
|---|---|---|---|
| coding, 98,808-token prompt | 96.9 tok/s | 99.0 tok/s | 98.7 tok/s |
| agentic, 100,043-token prompt | 89.8 tok/s | 90.7 tok/s | 91.2 tok/s |

All three runs wrote the same tokens. Peak VRAM 18.4 GB.

Known issue at release, fixed in v0.5.4:
- With `--parallel 3` or more and the MTP drafter on, the server could stop with `failed to remove sequence N`.
  See the [v0.5.4 release notes](../llamampere-v0.5.4/RELEASE_NOTES.md).
