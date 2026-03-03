"""
Shim package for the in-repo `aisuite` source tree.

This repository vendors `aisuite` under `aisuite/aisuite/`. When the repo root is on
`PYTHONPATH`, Python would otherwise treat `aisuite/` (without an `__init__.py`) as a
namespace package, which breaks absolute imports inside aisuite (e.g. `aisuite.framework.*`)
and callers that expect `aisuite.Client` to exist.

By providing this lightweight shim, we make `import aisuite` behave like a normal package
and extend its `__path__` to include the embedded implementation directory.
"""

from __future__ import annotations

from pathlib import Path
import pkgutil

# Make this a normal package and also look inside the embedded implementation directory.
__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_impl_dir = Path(__file__).resolve().parent / "aisuite"
if _impl_dir.exists():
    __path__.append(str(_impl_dir))

# Re-export the public surface that the backend expects.
try:
    from .client import Client  # noqa: F401
    from .framework.message import Message  # noqa: F401
    from .utils.tools import Tools  # noqa: F401
except Exception:
    # Allow partial imports in environments where dependencies aren't installed yet.
    pass

