# patch_imports.py - Apply patches to fix import issues before imports happen

import sys
import importlib.util
from importlib.machinery import ModuleSpec
from types import ModuleType

# Check if the module needs patching
def needs_patching():
    try:
        import huggingface_hub.snapshot_download
        return False  # No need to patch
    except ImportError:
        return True

# Only apply patch if needed
if needs_patching():
    # Create a virtual module with our compatibility code
    hf_compat_spec = ModuleSpec("huggingface_hub.snapshot_download", None)
    hf_compat_module = ModuleType(hf_compat_spec.name)
    sys.modules[hf_compat_spec.name] = hf_compat_module
    
    # Add REPO_ID_SEPARATOR to the virtual module
    hf_compat_module.REPO_ID_SEPARATOR = "--"
    
    print("Applied import patch for huggingface_hub.snapshot_download")
