# ffxiv-kbtools — the shared FFXIV file-access layer

The one copy of the modules more than one project imports. Copied here 2026-09-08 from
`learningHLSL\ffxiv-clarity-kb\tools\` (the working grab-bag), which stays where it is until the
`ffxiv-clarity-kb` split is finished; from now on **edit here**, and re-copy outward if the old
location is still in use.

| module | what |
|---|---|
| `sqpack.py` | index/dat reader, `find_game()`, `game_version()`, `entry_fingerprint()` (hash of the compressed on-disk entry) |
| `exdfile.py` | EXH/EXD reader |
| `texfile.py` | `.tex` header and mip layout |
| `texdecode.py` | BC1–BC7 / uncompressed → RGBA (`texture2ddecoder`, Python ≤ 3.12) |
| `texwrite.py` | `.tex` writer; BC7 via `bc7enc.py` when texconv is not available |
| `bc7enc.py` | numpy BC7 encoder (slow; tests and fallback) |
| `mtrlfile.py`, `imcfile.py`, `mdlstrings.py`, `mdlpatch.py` | material / imc / model-string readers |

Consumers:

* `ffxiv-texture-upscale\scripts-local\clarity-upscale` — `clarity\kbtools.py` looks here by
  default (`CLARITY_KB_TOOLS` overrides).
* `ffxiv-patchday\scripts-local` — carries its own `sqpack.py` / `exdfile.py`, byte-identical on
  2026-09-08 (sha256 `2ffc175a…`, `e3c6befe…`). They should import from here instead; until then,
  a change to either must be copied to both.

No `pyproject.toml`: these are plain modules put on `sys.path` by the consumer. Python 3.11–3.12
(`texture2ddecoder` has no 3.13/3.14 wheels).
