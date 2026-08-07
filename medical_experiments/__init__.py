"""Diagnostic experiments for medical LafitE/MLLM research.

The package deliberately keeps model extraction separate from hypothesis tests:
expensive backbones are run once, their features are cached, and the three
diagnostic gates operate on those immutable caches.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
