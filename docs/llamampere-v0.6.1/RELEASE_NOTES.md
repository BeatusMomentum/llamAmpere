# llamAmpere v0.6.1

A crash-fix release on top of v0.6. There are no new features and no new flags, and no speed change.

- **The 12 GB launch command no longer stops at startup.** In v0.6, running the README's 12 GB command (it sets
  `LLAMA_KQ_MASK_PACKED=1` and uses the SJ-KVaRN cache) made the server stop right after loading the model with a
  failed internal check on the attention mask width. The cause was a check that compared the packed mask against the
  size of the whole SJ-KVaRN cache buffer, but a given step only reads the part of the cache that is in use, and the
  mask is built for that smaller part. The check now compares the mask against the part that is actually read. Plain
  (non-SJ-KVaRN) caches keep an equivalent check. Without `LLAMA_KQ_MASK_PACKED=1` the crash never happened, so
  `LLAMA_KQ_MASK_PACKED=0` was a workaround on v0.6.

What stays the same:
- At temperature 0, the 24 GB README command (Swift 1.5 IQ4_XS-M, SJ-KVaRN 4/4) writes the same text as v0.6, byte
  for byte.
- With the 12 GB command, packed and unpacked masks give the same text.
- No GPU kernels changed.
