"""Code-owned per-model configs, selected by a short profile key.

Switching the served model is just picking a profile: on the server the key
rides in as the systemd template instance (``platform-ai-service@gemma`` ->
``Environment=MODEL_PROFILE=%i``), locally you can set ``MODEL_PROFILE`` in the
environment. ``applyProfile()`` seeds ``os.environ`` for the handful of per-model
vars *before* ``settings.py`` reads them, so a profile cleanly overrides any
stale ``.env`` values with no other code changes.

This module is the single source of truth for the configs documented in
MODELS.md. It imports only the stdlib (no ``import settings``) so it can run at
the very top of settings.py without a cycle.
"""

import os
import shlex
import sys

# Applied when MODEL_PROFILE is unset (e.g. a bare local run). The systemd units
# always pass an explicit instance, so this only matters off-server.
DEFAULT_PROFILE = "granite"

# The per-model vars that actually change between models. Every profile defines
# all of them, so switching always *resets* the full set — no value from a
# previously-served model can leak through. Everything else (dtype, quant,
# gpu-mem, tensor-parallel, max-num-seqs, HTTP/auth/HF) is shared and stays in
# .env. These names match the reads in settings.py exactly.
CORE_KEYS = (
    "MODEL",
    "SERVED_MODEL_NAME",
    "MAX_MODEL_LEN",
    "ENFORCE_EAGER",
    "TRUST_REMOTE_CODE",
    "ENABLE_AUTO_TOOL_CHOICE",
    "TOOL_CALL_PARSER",
    "VLLM_EXTRA_ARGS",
)

# Keys that mean "don't apply any profile — serve straight from raw .env". Lets
# you run a one-off model that isn't in the registry (start
# ``platform-ai-service@env`` and put its values in .env).
_PASSTHROUGH_KEYS = ("", "env", "custom", "none")

# ---------------------------------------------------------------------------
# The registry. Values mirror MODELS.md verbatim — update both together.
# VLLM_EXTRA_ARGS is a *list of tokens* (not a string): applyProfile joins it
# with shlex.join, which round-trips exactly through the shlex.split in
# ai/llm/VLLM.py, so embedded JSON like {"enable_thinking":false} reaches vLLM
# as one intact argv token (the old .env single-quoting footgun is gone).
# ---------------------------------------------------------------------------
# Currently EXPOSED profiles. To expose another, move its entry from
# _UNEXPOSED_PROFILES below into this dict (only PROFILES is selectable).
PROFILES = {
    # Dense, non-reasoning, compressed-tensors FP8 -> Cutlass fast path.
    # ~24 GB -> runs the full 128K context. Cleanest / known-good default.
    "granite": {
        "MODEL": "ibm-granite/granite-4.1-30b-fp8",
        "SERVED_MODEL_NAME": "granite-4.1-30b",
        "MAX_MODEL_LEN": 65536,
        "ENFORCE_EAGER": False,      # dense -> CUDA graphs on
        "TRUST_REMOTE_CODE": False,
        "ENABLE_AUTO_TOOL_CHOICE": False,
        "TOOL_CALL_PARSER": "",
        "VLLM_EXTRA_ARGS": [],
    },
    # FP8 -> Cutlass. Multimodal checkpoint; runs text-only (no images sent).
    # ENFORCE_EAGER=true (shared, in .env) is correct — gemma4 forces Triton attn.
    "gemma": {
        "MODEL": "RedHatAI/gemma-4-31B-it-FP8-dynamic",
        "SERVED_MODEL_NAME": "gemma-4-31b",
        "MAX_MODEL_LEN": 32768,
        "ENFORCE_EAGER": True,       # gemma4 forces the Triton attn backend
        "TRUST_REMOTE_CODE": False,
        "ENABLE_AUTO_TOOL_CHOICE": False,
        "TOOL_CALL_PARSER": "",
        "VLLM_EXTRA_ARGS": [],
    },
    # Native MXFP4 -> Marlin fallback on sm_120 (the "MARLIN Mxfp4" / "no native
    # FP4" logs are EXPECTED). Harmony reasoning parser applied by default ->
    # content already clean. NOTE: currently deleted from disk (~63 GB re-download).
    "oss120": {
        "MODEL": "openai/gpt-oss-120b",
        "SERVED_MODEL_NAME": "gpt-oss-120b",
        "MAX_MODEL_LEN": 32768,
        "ENFORCE_EAGER": False,
        "TRUST_REMOTE_CODE": False,
        "ENABLE_AUTO_TOOL_CHOICE": True,
        "TOOL_CALL_PARSER": "openai",
        "VLLM_EXTRA_ARGS": [],
    },
}

# Tested-and-working configs kept in code but NOT currently exposed. To enable
# one, move its entry into PROFILES above (values are ready to go). See MODELS.md
# for the per-model gotchas.
_UNEXPOSED_PROFILES = {
    # Instruct-2507 = no <think> blocks, so no reasoning parser needed; content
    # is clean. compressed-tensors FP8 -> Cutlass. For tools, set
    # TOOL_CALL_PARSER=hermes + ENABLE_AUTO_TOOL_CHOICE=True.
    "qwen3": {
        "MODEL": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "SERVED_MODEL_NAME": "qwen3-30b-a3b",
        "MAX_MODEL_LEN": 32768,
        "ENFORCE_EAGER": False,
        "TRUST_REMOTE_CODE": False,
        "ENABLE_AUTO_TOOL_CHOICE": False,
        "TOOL_CALL_PARSER": "",
        "VLLM_EXTRA_ARGS": [],
    },
    # Reasoning model. --reasoning-parser keeps CoT out of content;
    # enable_thinking:false defaults thinking OFF (clients can re-enable per
    # request). SLOW first load (GDN + MoE Triton compile) -> keep
    # BACKEND_STARTUP_TIMEOUT high in .env for the first boot.
    "qwen36": {
        "MODEL": "Qwen/Qwen3.6-35B-A3B-FP8",
        "SERVED_MODEL_NAME": "qwen3.6-35b",
        "MAX_MODEL_LEN": 32768,
        "ENFORCE_EAGER": True,       # GDN + MoE Triton compile — eager safer
        "TRUST_REMOTE_CODE": False,
        "ENABLE_AUTO_TOOL_CHOICE": False,
        "TOOL_CALL_PARSER": "",
        "VLLM_EXTRA_ARGS": [
            "--reasoning-parser", "qwen3",
            "--default-chat-template-kwargs", '{"enable_thinking":false}',
        ],
    },
    # Hybrid Mamba-2 + MoE. TRUST_REMOTE_CODE required (custom nemotron_h arch).
    # ModelOpt FP8 -> FlashInfer JIT (needs the CUDA-13 toolkit, in place).
    "nemotron": {
        "MODEL": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "SERVED_MODEL_NAME": "nemotron-3-nano",
        "MAX_MODEL_LEN": 32768,
        "ENFORCE_EAGER": True,       # Mamba-2 warmup — eager safer
        "TRUST_REMOTE_CODE": True,
        "ENABLE_AUTO_TOOL_CHOICE": True,
        "TOOL_CALL_PARSER": "qwen3_coder",
        "VLLM_EXTRA_ARGS": [
            "--reasoning-parser", "qwen3",
            "--default-chat-template-kwargs", '{"enable_thinking":false}',
        ],
    },
    # Same family as 120B, smaller — a fast GPT-OSS smoke test.
    # NOTE: currently deleted from disk (~16 GB re-download).
    "oss20": {
        "MODEL": "openai/gpt-oss-20b",
        "SERVED_MODEL_NAME": "gpt-oss-20b",
        "MAX_MODEL_LEN": 32768,
        "ENFORCE_EAGER": False,
        "TRUST_REMOTE_CODE": False,
        "ENABLE_AUTO_TOOL_CHOICE": True,
        "TOOL_CALL_PARSER": "openai",
        "VLLM_EXTRA_ARGS": [],
    },
}


def resolveKey() -> str:
    """The selected profile key (from MODEL_PROFILE, else the default)."""
    return (os.getenv("MODEL_PROFILE") or DEFAULT_PROFILE).strip().lower()


def envForProfile(key: str) -> dict:
    """The os.environ overrides a profile applies. Raises on an unknown key.

    Bools become "true"/"false" (settings._get_bool) and the VLLM_EXTRA_ARGS
    token list becomes a single shlex-joined string (settings + VLLM.py split it
    back apart). Empty strings are intentional: settings.py turns them into
    "flag omitted".
    """
    if key not in PROFILES:
        raise SystemExit(
            f"[profiles] unknown MODEL_PROFILE '{key}'. "
            f"Known: {', '.join(sorted(PROFILES))}"
        )
    profile = PROFILES[key]
    env = {}
    for var in CORE_KEYS:
        value = profile.get(var, "")
        if var == "VLLM_EXTRA_ARGS" and isinstance(value, (list, tuple)):
            value = shlex.join(value)
        elif isinstance(value, bool):
            value = "true" if value else "false"
        env[var] = str(value)
    return env


def applyProfile() -> None:
    """Seed os.environ from the selected profile, before settings.py reads it."""
    key = resolveKey()
    if key in _PASSTHROUGH_KEYS:
        return  # escape hatch: serve straight from raw .env
    for var, value in envForProfile(key).items():
        os.environ[var] = value


# ---------------------------------------------------------------------------
# Tiny operational CLI: `python profiles.py list|show <key>|validate <key>`.
# ---------------------------------------------------------------------------
def _main(argv) -> int:
    cmd = argv[0] if argv else "list"

    if cmd == "list":
        active = resolveKey()
        for key in sorted(PROFILES):
            marker = " *" if key == active else "  "
            print(f"{marker} {key:<10} {PROFILES[key]['MODEL']}")
        print(f"\nactive (MODEL_PROFILE or default): {active}")
        return 0

    if cmd == "show" and len(argv) > 1:
        for var, value in envForProfile(argv[1].strip().lower()).items():
            print(f"{var}={value}")
        return 0

    if cmd == "validate" and len(argv) > 1:
        key = argv[1].strip().lower()
        if key in _PASSTHROUGH_KEYS or key in PROFILES:
            return 0
        sys.stderr.write(
            f"[profiles] unknown profile '{key}'. "
            f"Known: {', '.join(sorted(PROFILES))}\n"
        )
        return 1

    sys.stderr.write("usage: python profiles.py list | show <key> | validate <key>\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
