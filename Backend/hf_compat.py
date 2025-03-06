# hf_compat.py - Compatibility layer for huggingface_hub
# This fixes the "No module named 'huggingface_hub.snapshot_download'" error

try:
    # Try the new import style
    from huggingface_hub import snapshot_download
    # Define the separator constant that sentence_transformers is looking for
    REPO_ID_SEPARATOR = "--"
except ImportError:
    try:
        # Try the old import style
        from huggingface_hub.snapshot_download import REPO_ID_SEPARATOR
    except ImportError:
        # Fallback definition if all else fails
        REPO_ID_SEPARATOR = "--"
