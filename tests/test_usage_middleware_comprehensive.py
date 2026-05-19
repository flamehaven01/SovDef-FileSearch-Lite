"""
Comprehensive tests for usage_middleware.py.
Tests _collect_exceeded_quotas (pure function) and
UsageTrackingMiddleware via ASGI test client.
"""

from flamehaven_filesearch.usage_middleware import (
    _collect_exceeded_quotas,
    UsageTrackingMiddleware,
)


# ---------------------------------------------------------------------------
# _collect_exceeded_quotas (pure helper)
# ---------------------------------------------------------------------------


class TestCollectExceededQuotas:
    def test_empty_status_returns_empty(self):
        result = _collect_exceeded_quotas({})
        assert result == []

    def test_daily_requests_exceeded(self):
        status = {
            "daily": {
                "requests": {"exceeded": True, "current": 1001, "limit": 1000},
                "tokens": {"exceeded": False, "current": 100, "limit": 1000000},
            },
            "monthly": {
                "requests": {"exceeded": False, "current": 500, "limit": 10000},
                "tokens": {"exceeded": False, "current": 1000, "limit": 5000000},
            },
        }
        result = _collect_exceeded_quotas(status)
        assert len(result) == 1
        assert "daily_requests" in result[0]
        assert "1001/1000" in result[0]

    def test_multiple_exceeded(self):
        status = {
            "daily": {
                "requests": {"exceeded": True, "current": 500, "limit": 100},
                "tokens": {"exceeded": True, "current": 999, "limit": 500},
            },
            "monthly": {
                "requests": {"exceeded": False, "current": 10, "limit": 10000},
                "tokens": {"exceeded": True, "current": 9999, "limit": 5000},
            },
        }
        result = _collect_exceeded_quotas(status)
        assert len(result) == 3

    def test_none_exceeded(self):
        status = {
            "daily": {
                "requests": {"exceeded": False, "current": 1, "limit": 1000},
                "tokens": {"exceeded": False, "current": 1, "limit": 1000},
            },
            "monthly": {
                "requests": {"exceeded": False, "current": 1, "limit": 10000},
                "tokens": {"exceeded": False, "current": 1, "limit": 100000},
            },
        }
        result = _collect_exceeded_quotas(status)
        assert result == []

    def test_partial_structure(self):
        status = {"daily": {"requests": {"exceeded": True, "current": 5, "limit": 3}}}
        result = _collect_exceeded_quotas(status)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# UsageTrackingMiddleware
# ---------------------------------------------------------------------------


class TestUsageTrackingMiddlewareDisabled:
    def test_disabled_middleware_passes_through(self):
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import PlainTextResponse
        from starlette.requests import Request

        async def homepage(request: Request):
            return PlainTextResponse("OK")

        app = Starlette(routes=[Route("/", homepage)])
        app.add_middleware(UsageTrackingMiddleware, enabled=False)

        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_non_api_path_passes_through(self):
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import PlainTextResponse
        from starlette.requests import Request

        async def health(request: Request):
            return PlainTextResponse("healthy")

        app = Starlette(routes=[Route("/health", health)])
        app.add_middleware(UsageTrackingMiddleware, enabled=True)

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_api_path_without_key_passes_through(self):
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.responses import PlainTextResponse
        from starlette.requests import Request

        async def search(request: Request):
            return PlainTextResponse("results")

        app = Starlette(routes=[Route("/api/search", search)])
        app.add_middleware(UsageTrackingMiddleware, enabled=True)

        client = TestClient(app)
        # No API key in headers; should pass through (no tracking)
        resp = client.get("/api/search")
        assert resp.status_code == 200
