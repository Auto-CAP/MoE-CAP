import pytest

from moe_cap.utils.report_metrics import calculate_wall_time_metrics


def test_wall_time_metrics_report_total_time_without_normalizing_by_request_count():
    one_request = calculate_wall_time_metrics(total_time=8.0, num_requests=1)
    many_requests = calculate_wall_time_metrics(total_time=8.0, num_requests=4)

    assert one_request["unnormalized_e2e"] == 8.0
    assert many_requests["unnormalized_e2e"] == 8.0
    assert many_requests["request/s"] == pytest.approx(0.5)
    assert "e2e_s" not in many_requests


def test_wall_time_metrics_handle_zero_elapsed_time_without_dividing_by_zero():
    metrics = calculate_wall_time_metrics(total_time=0.0, num_requests=4)

    assert metrics == {
        "unnormalized_e2e": 0.0,
        "request/s": 0.0,
    }


def test_wall_time_metrics_count_all_attempted_requests():
    metrics = calculate_wall_time_metrics(total_time=2.5, num_requests=10)

    assert metrics["request/s"] == pytest.approx(4.0)
