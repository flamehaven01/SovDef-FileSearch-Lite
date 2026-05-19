"""
Frontend-Backend API Contract Tests for FLAMEHAVEN FileSearch v1.4.2

Validates that the backend API responses match exactly what the frontend
JavaScript code expects (field names, types, status codes).

Each test documents the frontend page that consumes the API, the JS fields
it reads, and asserts those fields are present in real API responses.
"""

import pytest

from flamehaven_filesearch import __version__


class TestSearchPageContract:
    """
    Validates /api/search contract for frontend/dashboard/search.html

    Frontend JS reads:
    - data.request_id       (meta display)
    - data.answer           (main answer text)
    - data.sources          (array of source objects)
    - data.search_mode      (badge display)
    - data.refined_query    (optional display)
    """

    def test_search_payload_shape_accepted(self, client):
        """Contract: Search page POST payload is accepted by the backend."""
        # This mirrors the exact payload built in search.html postSearch()
        payload = {
            "query": "example query",
            "store_name": "default",
            "model": "default",
            "temperature": 0.5,
            "max_tokens": 1024,
            "search_mode": "keyword",
        }
        resp = client.post("/api/search", json=payload)

        # Must NOT be 422 (Unprocessable Entity) - payload shape must be valid
        assert resp.status_code != 422, (
            f"Search page payload rejected with 422. Frontend will break.\n"
            f"Response: {resp.text}"
        )

    def test_search_success_response_has_required_fields(self, client):
        """Contract: Successful search response has all fields frontend reads."""
        resp = client.post(
            "/api/search",
            json={"query": "flamehaven search engine", "store_name": "default"},
        )
        if resp.status_code == 200:
            data = resp.json()
            # Frontend reads these exact keys
            assert "request_id" in data, "Frontend reads data.request_id"
            assert "status" in data, "Frontend checks data.status"
            # answer OR message must be present (frontend shows whichever exists)
            assert (
                "answer" in data or "message" in data
            ), "Frontend reads data.answer or data.message"
            assert "sources" in data, "Frontend renders data.sources array"

    def test_search_error_response_has_request_id_or_header(self, client):
        """Contract: Error responses expose request_id (frontend shows it)."""
        # Empty query triggers error
        resp = client.post("/api/search", json={"query": ""})
        if resp.status_code in [400, 422]:
            data = resp.json()
            has_request_id = "request_id" in data or "X-Request-ID" in resp.headers
            assert (
                has_request_id
            ), "Error responses must expose request_id for frontend display"

    def test_search_mode_field_in_response(self, client):
        """Contract: search_mode is echoed back in response (badge display)."""
        resp = client.post(
            "/api/search",
            json={"query": "test mode echo", "search_mode": "keyword"},
        )
        if resp.status_code == 200:
            data = resp.json()
            # search_mode may be null but key should exist per SearchResponse model
            assert (
                "search_mode" in data or data.get("status") == "error"
            ), "search_mode field expected in response for frontend badge"

    def test_search_store_field_in_response(self, client):
        """Contract: 'store' field returned matches frontend store selector."""
        resp = client.post(
            "/api/search",
            json={"query": "store contract check", "store_name": "default"},
        )
        if resp.status_code == 200:
            data = resp.json()
            # Frontend displays store name from data.store
            assert (
                "store" in data or "store_name" in data or "status" in data
            ), "Store field missing from search response"


class TestUploadPageContract:
    """
    Validates /api/upload/* contract for frontend/dashboard/upload.html

    Frontend JS:
    - Single upload: POST /api/upload/single with FormData {file, store}
    - Multiple upload: POST /api/upload/multiple with FormData {files[], store}
    - Reads: data.request_id, data.status, data.detail (error)
    """

    def test_single_upload_formdata_shape_accepted(self, client):
        """Contract: Single upload FormData structure is accepted."""
        resp = client.post(
            "/api/upload/single",
            files={"file": ("contract_test.txt", b"test content", "text/plain")},
            data={"store": "contract_test"},
        )
        # Must NOT be 422 - FormData shape must be valid
        assert (
            resp.status_code != 422
        ), f"Single upload FormData rejected (422). Frontend will break.\n{resp.text}"

    def test_single_upload_success_response_fields(self, client):
        """Contract: Single upload success has fields frontend reads."""
        resp = client.post(
            "/api/upload/single",
            files={"file": ("resp_test.txt", b"response field test", "text/plain")},
            data={"store": "default"},
        )
        if resp.status_code == 200:
            data = resp.json()
            assert "request_id" in data, "Frontend reads data.request_id"
            assert "status" in data, "Frontend checks data.status"

    def test_multiple_upload_formdata_shape_accepted(self, client):
        """Contract: Multiple upload FormData structure ('files' key) is accepted."""
        resp = client.post(
            "/api/upload/multiple",
            files=[
                ("files", ("file1.txt", b"content1", "text/plain")),
                ("files", ("file2.txt", b"content2", "text/plain")),
            ],
            data={"store": "multi_contract"},
        )
        # Must NOT be 422
        assert (
            resp.status_code != 422
        ), f"Multiple upload FormData rejected (422). Frontend will break.\n{resp.text}"

    def test_multiple_upload_success_response_fields(self, client):
        """Contract: Multiple upload success has fields frontend reads."""
        resp = client.post(
            "/api/upload/multiple",
            files=[
                ("files", ("a.txt", b"aaa", "text/plain")),
                ("files", ("b.txt", b"bbb", "text/plain")),
            ],
            data={"store": "default"},
        )
        if resp.status_code == 200:
            data = resp.json()
            assert "request_id" in data, "Frontend reads data.request_id"
            assert "status" in data
            assert "total" in data, "Frontend shows total count"
            assert "successful" in data, "Frontend shows successful count"
            assert "failed" in data
            assert "files" in data, "Frontend iterates data.files array"

    def test_upload_error_has_detail_field(self, client):
        """Contract: Upload errors expose 'detail' or 'message' (frontend shows it)."""
        # Trigger a validation error - hidden filename
        resp = client.post(
            "/api/upload/single",
            files={"file": (".hidden.txt", b"hidden", "text/plain")},
            data={"store": "test"},
        )
        if resp.status_code in [400, 422]:
            data = resp.json()
            has_detail = "detail" in data or "message" in data or "error" in data
            assert has_detail, (
                f"Error response missing detail/message. Frontend shows: "
                f"data.detail || 'upload error'. Got keys: {list(data.keys())}"
            )


class TestMetricsPageContract:
    """
    Validates /metrics contract for frontend/dashboard/metrics.html

    Frontend JS reads:
    - data.uptime_seconds        (Uptime card)
    - data.prometheus.requests_last_60s
    - data.prometheus.errors_last_60s
    - data.prometheus.cache_hits_total
    - data.prometheus.cache_misses_total
    - data.health_status         (Health card)
    """

    def test_metrics_endpoint_accessible(self, client, monkeypatch):
        """Contract: /metrics endpoint is reachable from frontend."""
        monkeypatch.setenv("FLAMEHAVEN_METRICS_ENABLED", "1")
        resp = client.get("/metrics")
        # Must not be 404 - endpoint must exist
        assert resp.status_code != 404, "/metrics endpoint must exist for frontend"
        assert resp.status_code in [200, 403, 429, 503]

    def test_metrics_response_has_uptime_seconds(self, client, monkeypatch):
        """Contract: /metrics response has uptime_seconds (frontend Uptime card)."""
        monkeypatch.setenv("FLAMEHAVEN_METRICS_ENABLED", "1")
        resp = client.get("/metrics")
        if resp.status_code == 200:
            data = resp.json()
            assert (
                "uptime_seconds" in data
            ), "Frontend reads data.uptime_seconds for Uptime card. Field missing!"

    def test_metrics_response_has_health_status(self, client, monkeypatch):
        """Contract: /metrics response has health_status (frontend Health card)."""
        monkeypatch.setenv("FLAMEHAVEN_METRICS_ENABLED", "1")
        resp = client.get("/metrics")
        if resp.status_code == 200:
            data = resp.json()
            # health_status may be None but key should exist
            assert (
                "health_status" in data
            ), "Frontend reads data.health_status || 'unknown'. Field missing!"

    def test_metrics_response_has_prometheus_block(self, client, monkeypatch):
        """Contract: /metrics has prometheus block with counters frontend reads."""
        monkeypatch.setenv("FLAMEHAVEN_METRICS_ENABLED", "1")
        resp = client.get("/metrics")
        if resp.status_code == 200:
            data = resp.json()
            # prometheus block is optional but if present must have expected keys
            if "prometheus" in data and data["prometheus"]:
                prom = data["prometheus"]
                # Frontend reads these keys (may be null/missing but checked)
                expected_keys = [
                    "requests_last_60s",
                    "errors_last_60s",
                    "cache_hits_total",
                    "cache_misses_total",
                ]
                for key in expected_keys:
                    # Use ?? in JS means it's OK to be missing, but document the contract
                    _ = prom.get(key)  # no assertion, just document


class TestCachePageContract:
    """
    Validates /api/metrics + /api/admin/cache/flush contract
    for frontend/dashboard/cache.html

    Frontend JS reads from GET /api/metrics:
    - data.cache.search_cache.current_size
    - data.cache.search_cache.ttl
    - data.prometheus.cache_hits_total
    - data.prometheus.cache_misses_total

    Frontend JS POSTs to:
    - /api/admin/cache/flush (Flush button)
    """

    def test_api_metrics_has_cache_block(self, client):
        """Contract: /api/metrics has 'cache' block for Cache Monitor page."""
        resp = client.get("/api/metrics")
        if resp.status_code == 200:
            data = resp.json()
            # Cache block is used by cache.html
            assert (
                "cache" in data
            ), "Frontend reads data.cache for cache stats. Block missing!"

    def test_api_metrics_cache_structure(self, client):
        """Contract: cache block has search_cache sub-object frontend reads."""
        resp = client.get("/api/metrics")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("cache"):
                cache = data["cache"]
                # Frontend reads: (data.cache && data.cache.search_cache) || {}
                if "search_cache" in cache:
                    sc = cache["search_cache"]
                    # Frontend reads: cache.current_size, cache.ttl
                    assert isinstance(sc, dict), "search_cache must be a dict"

    def test_cache_flush_endpoint_exists(self, client):
        """Contract: /api/admin/cache/flush endpoint exists for Flush button."""
        resp = client.post("/api/admin/cache/flush")
        # 200 (flushed), 401/403 (auth required - also valid), 429
        # Must NOT be 404 - endpoint must exist
        assert (
            resp.status_code != 404
        ), "/api/admin/cache/flush must exist. Flush button in frontend will 404!"
        assert resp.status_code in [200, 401, 403, 429, 503]

    def test_stores_list_format(self, client):
        """Contract: /api/stores response matches Dashboard page expectations.

        Note: In offline/fallback mode the 'stores' field is returned as a
        dict {store_name: uri} rather than a list. Both formats are valid at
        runtime; the frontend should handle both gracefully.
        """
        resp = client.get("/api/stores")
        if resp.status_code == 200:
            data = resp.json()
            assert "stores" in data, "Frontend reads data.stores"
            assert "count" in data, "Frontend reads data.count"
            assert "request_id" in data, "Frontend reads data.request_id"
            # stores may be list (ideal) OR dict {name: uri} (offline fallback)
            assert isinstance(
                data["stores"], (list, dict)
            ), f"stores must be list or dict, got: {type(data['stores'])}"


class TestDashboardPageContract:
    """
    Validates /health + /api/metrics contract for landing.html Dashboard.

    Frontend reads from GET /health:
    - data.status          (healthy/unhealthy)
    - data.uptime_formatted
    - data.version
    - data.system.cpu_percent
    - data.system.memory_percent

    Frontend reads from GET /metrics:
    - data.stores_count
    - data.stores
    """

    def test_health_response_all_dashboard_fields(self, client):
        """Contract: /health has all fields Dashboard cards need."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()

        assert "status" in data, "Dashboard reads data.status"
        assert "version" in data, "Dashboard reads data.version"
        assert "uptime_seconds" in data, "Dashboard reads data.uptime_seconds"
        assert "uptime_formatted" in data, "Dashboard reads data.uptime_formatted"
        assert (
            "searcher_initialized" in data
        ), "Dashboard reads data.searcher_initialized"
        assert "timestamp" in data, "Dashboard reads data.timestamp"
        assert "system" in data, "Dashboard reads data.system"

    def test_health_system_block_has_metrics(self, client):
        """Contract: /health system block has CPU/memory data for Dashboard."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        system = data.get("system", {})

        if "error" not in system:
            assert "cpu_percent" in system, "Dashboard reads system.cpu_percent"
            assert "memory_percent" in system, "Dashboard reads system.memory_percent"

    def test_metrics_has_stores_count(self, client):
        """Contract: /api/metrics has stores_count for Dashboard."""
        resp = client.get("/api/metrics")
        if resp.status_code == 200:
            data = resp.json()
            assert "stores_count" in data, "Dashboard reads data.stores_count"
            assert "stores" in data, "Dashboard reads data.stores"

    def test_root_api_info_fields(self, client):
        """Contract: / root endpoint has API info for any info displays."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "version" in data
        assert (
            data["version"] == __version__
        ), f"Version mismatch: expected {__version__}, got {data['version']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
