"""VoiceClip local web viewer — HTTP server + routes + static assets.

This package contains the full viewer subsystem:

  server.py           — Handler class (HTTP plumbing + dispatch) and serve()
  routes/             — one module per API feature domain (entries,
                        research, patterns, summaries, settings, models,
                        consent). Each registers handlers into a shared
                        dispatch registry.
  static/             — frontend CSS, JS, HTML, jsconfig.json

Public API — re-exported here so `from voiceclip.viewer import Handler`
and `from voiceclip.viewer import serve` continue working identically to
when viewer was a flat module. Internal consumers (tests, __main__.py)
don't need to know the package shape changed.
"""

from voiceclip.viewer.server import Handler, serve

__all__ = ["Handler", "serve"]
