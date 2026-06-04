# -------------------- Default model identifiers -------------------- #
# Convenience constants for common HuggingFace model ids. The actual model is
# chosen at runtime via the MODEL env var (see settings.py); these are just
# documented, sensible defaults a deployer can point MODEL at.

# Small, ungated — good for a first run / smoke test on modest GPUs.
Qwen25_05B = "Qwen/Qwen2.5-0.5B-Instruct"
Qwen25_15B = "Qwen/Qwen2.5-1.5B-Instruct"
Qwen25_7B = "Qwen/Qwen2.5-7B-Instruct"

# Larger / gated — require HF_TOKEN and more VRAM.
Llama31_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
Llama33_70B = "meta-llama/Llama-3.3-70B-Instruct"

# Default served model when MODEL is not provided.
DEFAULT_MODEL = Qwen25_15B

# -------------------- Misc -------------------- #
# Role used by the chat handler for assistant turns (OpenAI convention).
RESPONSE_ROLE = "assistant"
