"""HTTP route modules for the VoiceClip viewer.

Each submodule registers its GET / POST handlers into the module-level
registries below by importing `register_get` / `register_post` and
calling them with a path and a handler function. The viewer's `Handler`
class looks up routes by path at dispatch time.

Handler signatures:

    def get_handler(req, query: dict[str, list[str]]) -> None:
        ...
        req._json(payload)

    def post_handler(req, payload: dict) -> None:
        ...
        req._json({"ok": True})

`req` is the `BaseHTTPRequestHandler` subclass (`voiceclip.viewer.Handler`).
Handlers call `req._json(...)`, `req._not_found()`, etc to respond —
they don't return values. This mirrors the prior inline shape so the
migration stays mechanical.

Why a registry instead of decorators on `Handler`: decorators on the
Handler class would require re-binding at import time and scatter routes
across modules that all need to touch the same class. A plain dict keyed
by path is simpler and matches how Handler already dispatches (string
match on `url.path`).
"""

from __future__ import annotations

from collections.abc import Callable

# Type alias for handler functions.  Both GET and POST handlers accept a
# request handler and a second arg (query dict or JSON payload). They
# respond by calling methods on the request handler; they don't return.
RouteHandler = Callable[..., None]

# Populated at import time by each route submodule.
_GET_ROUTES: dict[str, RouteHandler] = {}
_POST_ROUTES: dict[str, RouteHandler] = {}


def register_get(path: str, handler: RouteHandler) -> None:
    """Register a GET handler for an exact path match."""
    if path in _GET_ROUTES:
        raise RuntimeError(f"duplicate GET route: {path}")
    _GET_ROUTES[path] = handler


def register_post(path: str, handler: RouteHandler) -> None:
    """Register a POST handler for an exact path match."""
    if path in _POST_ROUTES:
        raise RuntimeError(f"duplicate POST route: {path}")
    _POST_ROUTES[path] = handler


def get_handler(path: str) -> RouteHandler | None:
    """Look up a GET handler by path. None if no match."""
    return _GET_ROUTES.get(path)


def post_handler(path: str) -> RouteHandler | None:
    """Look up a POST handler by path. None if no match."""
    return _POST_ROUTES.get(path)


def load_all() -> None:
    """Import every route submodule so they populate the registries.

    Called once by `voiceclip.viewer` at import time. Idempotent — if the
    registries already have entries, this is a no-op. Keeping this
    explicit (rather than relying on side-effects of any random import)
    makes the startup trace clearer and catches import errors at server
    start rather than on first request.
    """
    # Idempotency guard: if routes are already loaded, don't re-import
    # (which would trigger duplicate-registration errors).
    if _GET_ROUTES or _POST_ROUTES:
        return

    # Import order doesn't matter — each module registers its own paths.
    # Listing them here makes it obvious which modules participate.
    from voiceclip.viewer.routes import (  # noqa: F401
        consent,
        entries,
        models,
        patterns,
        research,
        settings,
        summaries,
    )
