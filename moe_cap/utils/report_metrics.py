"""Pure helpers for reporting per-request timing / throughput metrics."""

from typing import Dict


def calculate_request_metrics(total_time: float, num_requests: int) -> Dict[str, float]:
    """Report normalized (per-request) end-to-end latency and throughput.

    Args:
        total_time: Total wall-clock time for the run.
        num_requests: Number of attempted requests (N), including failures.

    Returns:
        A dict with exactly two keys:
        - "e2e_s": total_time / N, the mean per-request end-to-end latency
          (0.0 when N <= 0).
        - "request/s": N / total_time, or 0.0 when total_time <= 0.
    """
    e2e_s = round(total_time / num_requests, 2) if num_requests > 0 else 0.0
    request_s = round(num_requests / total_time, 4) if total_time > 0 else 0.0
    return {"e2e_s": e2e_s, "request/s": request_s}
