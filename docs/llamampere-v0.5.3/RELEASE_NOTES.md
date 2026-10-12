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
  (one GPU file, `mmvq.cu`). Fixed; the RTX 3090 build is unchanged (identical GPU machine code). 40- and 50-series
  cards are not tuned for, see the build note in the README.
- `--parallel 1` decode speed at 100K depth is unchanged from v0.5.2.
