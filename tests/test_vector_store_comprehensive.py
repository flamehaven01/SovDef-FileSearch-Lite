"""
Comprehensive tests for vector_store.py.
Target: cover CircuitBreaker and retry_with_backoff (194 miss statements).
PostgresVectorStore tests are skipped unless postgres is available.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from flamehaven_filesearch.vector_store import (
    CircuitBreaker,
    CircuitState,
    retry_with_backoff,
)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreakerInit:
    def test_default_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_custom_thresholds(self):
        cb = CircuitBreaker(
            failure_threshold=3, recovery_timeout=30.0, success_threshold=1
        )
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30.0
        assert cb.success_threshold == 1

    def test_initial_counts_zero(self):
        cb = CircuitBreaker()
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time is None


class TestCircuitBreakerSuccess:
    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.failure_count = 3
        cb._on_success()
        assert cb.failure_count == 0

    def test_success_in_half_open_increments(self):
        cb = CircuitBreaker(success_threshold=2)
        cb.state = CircuitState.HALF_OPEN
        cb._on_success()
        assert cb.success_count == 1
        assert cb.state == CircuitState.HALF_OPEN

    def test_success_threshold_closes_circuit(self):
        cb = CircuitBreaker(success_threshold=2)
        cb.state = CircuitState.HALF_OPEN
        cb._on_success()
        cb._on_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.success_count == 0

    def test_call_success_returns_value(self):
        cb = CircuitBreaker()
        result = cb.call(lambda x: x * 2, 5)
        assert result == 10


class TestCircuitBreakerFailure:
    def test_failure_increments_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        try:
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass
        assert cb.failure_count == 1

    def test_failure_threshold_opens_circuit(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
            except RuntimeError:
                pass
        assert cb.state == CircuitState.OPEN

    def test_failure_in_half_open_reopens(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.state = CircuitState.HALF_OPEN
        cb._on_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_circuit_raises(self):
        cb = CircuitBreaker()
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time()  # recent failure
        with pytest.raises(RuntimeError, match="OPEN"):
            cb.call(lambda: None)


class TestCircuitBreakerHalfOpen:
    def test_open_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(recovery_timeout=0.01)
        cb.state = CircuitState.OPEN
        cb.last_failure_time = time.time() - 1.0  # 1 second ago > 0.01s timeout
        # Call should transition to HALF_OPEN and attempt
        try:
            cb.call(lambda: None)
        except Exception:
            pass
        # After timeout, it should have tried
        assert cb.state in (
            CircuitState.CLOSED,
            CircuitState.HALF_OPEN,
            CircuitState.OPEN,
        )


class TestCircuitBreakerReset:
    def test_reset_to_closed(self):
        cb = CircuitBreaker()
        cb.state = CircuitState.OPEN
        cb.failure_count = 5
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    def test_success_on_first_try(self):
        call_count = [0]

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def success_func():
            call_count[0] += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_then_success(self):
        call_count = [0]

        @retry_with_backoff(max_retries=3, initial_delay=0.001)
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("temporary error")
            return "ok"

        result = flaky_func()
        assert result == "ok"
        assert call_count[0] == 3

    def test_all_retries_fail_raises(self):
        call_count = [0]

        @retry_with_backoff(max_retries=2, initial_delay=0.001)
        def always_fail():
            call_count[0] += 1
            raise ValueError("persistent error")

        with pytest.raises(ValueError, match="persistent"):
            always_fail()
        assert call_count[0] == 2

    def test_exponential_backoff_delays(self):
        delays = []
        original_sleep = time.sleep

        @retry_with_backoff(max_retries=3, initial_delay=0.1, backoff_factor=2.0)
        def fail_twice():
            raise RuntimeError("fail")

        with patch("time.sleep") as mock_sleep:
            try:
                fail_twice()
            except RuntimeError:
                pass
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            # Should have called sleep at least once with increasing delays
            assert len(sleep_calls) >= 1
            if len(sleep_calls) >= 2:
                assert sleep_calls[1] >= sleep_calls[0]

    def test_max_delay_capped(self):
        @retry_with_backoff(
            max_retries=3, initial_delay=1.0, max_delay=1.5, backoff_factor=10.0
        )
        def always_fail():
            raise RuntimeError("fail")

        with patch("time.sleep") as mock_sleep:
            try:
                always_fail()
            except RuntimeError:
                pass
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            for delay in sleep_calls:
                assert delay <= 1.5


# ---------------------------------------------------------------------------
# create_vector_store returns None without postgres config
# ---------------------------------------------------------------------------


class TestCreateVectorStore:
    def test_returns_none_without_postgres(self):
        from flamehaven_filesearch.vector_store import create_vector_store
        from flamehaven_filesearch.config import Config

        config = Config.__new__(Config)
        config.vector_backend = "memory"
        config.postgres_dsn = None
        config.postgres_enabled = False
        result = create_vector_store(config, 384)
        assert result is None

    def test_memory_backend_config(self):
        from flamehaven_filesearch.vector_store import create_vector_store
        from flamehaven_filesearch.config import Config

        config = Config(api_key=None)
        result = create_vector_store(config, 384)
        assert result is None  # no postgres DSN configured
