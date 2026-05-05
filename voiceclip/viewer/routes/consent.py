"""Cloud-provider consent endpoints.

GET  /api/consent      — pending cloud features (+ what data goes where)
POST /api/consent/ack  — user dismissed the in-UI banner, record ack
"""

from __future__ import annotations

from voiceclip.consent import FEATURE_DATA_SENT, pending_acks, record_acks
from voiceclip.viewer.routes import register_get, register_post


def _get_consent(req, query):
    pend = pending_acks()
    out = {
        feature: {
            **info,
            "data_sent": FEATURE_DATA_SENT.get(feature, ""),
        }
        for feature, info in pend.items()
    }
    req._json({"pending": out})


def _post_ack(req, payload):
    features = payload.get("features")
    if features is not None and not isinstance(features, list):
        req._json({"error": "features must be a list or omitted"}, status=400)
        return
    record_acks(features=features)
    req._json({"ok": True})


register_get("/api/consent", _get_consent)
register_post("/api/consent/ack", _post_ack)
