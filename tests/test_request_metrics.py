import pytest

from moe_cap.utils.report_metrics import calculate_request_metrics


def test_e2e_is_normalized_per_request_and_request_throughput():
    one_request = calculate_request_metrics(total_time=8.0, num_requests=1)
    many_requests = calculate_request_metrics(total_time=8.0, num_requests=4)

    # e2e_s is the per-request mean (total_time / N), not the raw wall clock.
    assert one_request["e2e_s"] == pytest.approx(8.0)
    assert many_requests["e2e_s"] == pytest.approx(2.0)
    assert many_requests["request/s"] == pytest.approx(0.5)
    # the raw/unnormalized key must not be emitted
    assert "unnormalized_e2e" not in many_requests


def test_handle_zero_elapsed_time_without_dividing_by_zero():
    metrics = calculate_request_metrics(total_time=0.0, num_requests=4)
    assert metrics == {"e2e_s": 0.0, "request/s": 0.0}


def test_request_throughput_counts_all_attempted_requests():
    metrics = calculate_request_metrics(total_time=2.5, num_requests=10)
    assert metrics["request/s"] == pytest.approx(4.0)
    assert metrics["e2e_s"] == pytest.approx(0.25)
