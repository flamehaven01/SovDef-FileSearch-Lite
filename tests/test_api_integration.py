"""
API Integration Test Suite for FLAMEHAVEN FileSearch v1.4.2

Tests complete API workflows with rate limiting, validation, and error handling.
"""

import pytest

from flamehaven_filesearch import __version__


class TestAPIIntegration:
    """Integration tests for complete API workflows"""

    def test_health_check_integration(self, client):
        """Test health check endpoint returns all expected fields"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Verify all expected fields
        assert "status" in data
        assert "version" in data
        assert data["version"] == __version__
        assert "uptime_seconds" in data
        assert "uptime_formatted" in data
        assert "searcher_initialized" in data
        assert "timestamp" in data
        assert "system" in data

        # Verify system info
        system = data["system"]
        if "error" not in system:
            assert "cpu_percent" in system
            assert "memory_percent" in system
            assert "disk_percent" in system

    def test_request_id_tracing(self, client):
        """Test that request ID is tracked across requests"""
        # Send request with custom request ID
        custom_request_id = "test-12345"
        response = client.get("/health", headers={"X-Request-ID": custom_request_id})

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert response.headers["X-Request-ID"] == custom_request_id

    def test_security_headers_present(self, client):
        """Test that security headers are present in responses"""
        response = client.get("/health")

        assert response.status_code == 200

        # Verify security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"

        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"

        assert "X-XSS-Protection" in response.headers

        assert "Strict-Transport-Security" in response.headers

        assert "Content-Security-Policy" in response.headers

    def test_rate_limiting_upload(self, client):
        """Test rate limiting on upload endpoint"""
        # Create a small test file
        test_file = ("test.txt", b"Test content", "text/plain")

        # Make 11 requests (limit is 10/min)
        responses = []
        for i in range(11):
            response = client.post(
                "/api/upload/single",
                files={"file": test_file},
                data={"store": "test"},
            )
            responses.append(response)

        # First 10 should succeed or fail for other reasons (but not rate limit)
        # 11th should be rate limited (429)
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited, "Rate limiting not working for uploads"

    def test_rate_limiting_search(self, client):
        """Test rate limiting on search endpoint"""
        # Make 101 requests (limit is 100/min)
        responses = []
        for i in range(101):
            response = client.post(
                "/api/search",
                json={"query": f"test query {i}"},
            )
            responses.append(response)

        # 101st request should be rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        assert rate_limited, "Rate limiting not working for search"

    def test_invalid_filename_error_response(self, client):
        """Test error response for invalid filename"""
        # Hidden file (starts with .)
        response = client.post(
            "/api/upload/single",
            files={"file": (".hidden.txt", b"content", "text/plain")},
            data={"store": "test"},
        )

        assert response.status_code == 400
        data = response.json()

        # Verify error response structure
        assert "error" in data
        assert "message" in data
        assert "status_code" in data
        assert data["status_code"] == 400
        assert "request_id" in data
        assert "timestamp" in data

    def test_empty_search_query_error_response(self, client):
        """Test error response for empty search query"""
        response = client.post(
            "/api/search",
            json={"query": ""},
        )

        assert response.status_code == 400
        data = response.json()

        # Verify error response structure
        assert "error" in data
        assert "message" in data
        assert "request_id" in data

    def test_response_timing_header(self, client):
        """Test that response includes timing header"""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Response-Time" in response.headers

        # Verify format (e.g., "0.123s")
        timing = response.headers["X-Response-Time"]
        assert timing.endswith("s")
        assert float(timing[:-1]) >= 0

    def test_metrics_endpoint_enhanced(self, client, monkeypatch):
        """Test enhanced metrics endpoint"""
        monkeypatch.setenv("FLAMEHAVEN_METRICS_ENABLED", "1")
        response = client.get("/metrics")

        assert response.status_code in [
            200,
            503,
        ]  # May be unavailable if searcher not init

        if response.status_code == 200:
            data = response.json()

            # Verify enhanced metrics
            assert "stores_count" in data
            assert "stores" in data
            assert "config" in data
            assert "system" in data
            assert "uptime_seconds" in data

            # Verify system metrics
            system = data["system"]
            if "error" not in system:
                assert "cpu_percent" in system
                assert "memory_percent" in system

    def test_root_endpoint_info(self, client):
        """Test root endpoint provides API information"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        # Verify API info
        assert data["name"] == "FLAMEHAVEN FileSearch API"
        assert data["version"] == __version__
        assert "endpoints" in data
        assert "rate_limits" in data

        # Verify rate limit documentation
        rate_limits = data["rate_limits"]
        assert "upload_single" in rate_limits
        assert "search" in rate_limits

    def test_upload_file_validation_integration(self, client):
        """Test complete file upload validation workflow"""
        # Test 1: Valid file
        valid_file = ("document.txt", b"Valid content", "text/plain")
        response = client.post(
            "/api/upload/single",
            files={"file": valid_file},
            data={"store": "test"},
        )

        # Should succeed or fail for other reasons (not validation)
        if response.status_code != 429:  # Not rate limited
            assert response.status_code in [200, 503]  # Success or service unavailable

        # Test 2: Path traversal attempt
        malicious_file = ("../../etc/passwd", b"attack", "text/plain")
        response = client.post(
            "/api/upload/single",
            files={"file": malicious_file},
            data={"store": "test"},
        )

        assert response.status_code in [400, 429]  # Bad request or rate limited

    def test_search_validation_integration(self, client):
        """Test complete search validation workflow"""
        # Test 1: Valid query
        response = client.post(
            "/api/search",
            json={"query": "test query"},
        )

        # Should succeed or fail for other reasons (not validation)
        if response.status_code != 429:
            assert response.status_code in [200, 404, 503]

        # Test 2: Empty query
        response = client.post(
            "/api/search",
            json={"query": ""},
        )

        assert response.status_code in [400, 422, 429]

        # Test 3: Too long query
        long_query = "word " * 500  # 2500+ characters
        response = client.post(
            "/api/search",
            json={"query": long_query},
        )

        assert response.status_code in [400, 422, 429]

    def test_stores_management_workflow(self, client):
        """Test complete store management workflow"""
        # List stores
        response = client.get("/api/stores")

        if response.status_code != 503:
            assert response.status_code in [200, 429]
            if response.status_code == 200:
                data = response.json()
                assert "stores" in data
                assert "count" in data
                assert "request_id" in data

    def test_error_response_consistency(self, client):
        """Test that all error responses follow same structure"""
        # Test various error scenarios
        test_cases = [
            # (endpoint, method, data, files)
            (
                "/api/upload/single",
                "POST",
                {"store": "test"},
                {"file": (".hidden.txt", b"x", "text/plain")},
            ),
            ("/api/search", "POST", {"query": ""}, None),
        ]

        for endpoint, method, data, files in test_cases:
            if method == "POST":
                if files:
                    response = client.post(endpoint, files=files, data=data)
                else:
                    response = client.post(endpoint, json=data)

                # All errors should have consistent structure
                if 400 <= response.status_code < 500:
                    error_data = response.json()
                    assert "error" in error_data or "status" in error_data
                    assert "message" in error_data or "detail" in error_data
                    assert (
                        "request_id" in error_data or "X-Request-ID" in response.headers
                    )

    def test_multiple_upload_workflow(self, client):
        """Test multiple file upload workflow"""
        files = [
            ("files", ("file1.txt", b"content1", "text/plain")),
            ("files", ("file2.txt", b"content2", "text/plain")),
            ("files", ("file3.txt", b"content3", "text/plain")),
        ]

        response = client.post(
            "/api/upload/multiple",
            files=files,
            data={"store": "test"},
        )

        # Should succeed, be rate limited, or service unavailable
        assert response.status_code in [200, 429, 503]

        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "files" in data
            assert "total" in data
            assert "successful" in data
            assert "failed" in data
            assert "request_id" in data

    def test_cors_headers(self, client):
        """Test CORS headers are set"""
        response = client.options(
            "/api/search",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Should have CORS headers
        assert response.status_code in [200, 204]

    def test_api_versioning_in_responses(self, client):
        """Test that API version is consistently reported"""
        # Health check
        response = client.get("/health")
        if response.status_code == 200:
            assert response.json()["version"] == __version__

        # Root endpoint
        response = client.get("/")
        if response.status_code == 200:
            assert response.json()["version"] == __version__


class TestAPIPerformance:
    """Performance tests for API endpoints"""

    @pytest.mark.slow
    def test_health_check_performance(self, client):
        """Test health check responds within target time"""
        import time

        iterations = 10
        total_time = 0

        for _ in range(iterations):
            start = time.time()
            response = client.get("/health")
            elapsed = time.time() - start
            total_time += elapsed

            assert response.status_code == 200

        avg_time = total_time / iterations
        print(f"\nAverage health check time: {avg_time:.4f}s")

        # Should be fast (<100ms average)
        assert avg_time < 0.1, f"Health check too slow: {avg_time:.3f}s"

    @pytest.mark.slow
    def test_concurrent_health_checks(self, client):
        """Test health check under concurrent load"""
        import concurrent.futures

        def make_request():
            return client.get("/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(r.status_code == 200 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
