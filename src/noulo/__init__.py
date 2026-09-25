"""noulo — local Choice / Score / Noul decision engine.

from noulo import evaluate
evaluate({"type": "noul", "input": "...", "proposition": "..."})
"""

from typing import Any

__version__ = "0.1.0"
__all__ = ["Noulo", "RequestError", "aevaluate", "close", "evaluate", "__version__"]


def __getattr__(name: str) -> Any:  # lazy: `import noulo` stays cheap for the CLI
    if name in ("Noulo", "evaluate", "aevaluate", "close"):
        from . import embedded

        return getattr(embedded, name)
    if name == "RequestError":
        from .api.validation import RequestError

        return RequestError
    raise AttributeError(f"module 'noulo' has no attribute {name!r}")
