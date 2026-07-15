"""Pure helpers for reporting wall-clock timing metrics."""

from typing import Dict


def calculate_wall_time_metrics(total_time: float, num_requests: int) -> Dict[str, float]:
    """Report raw wall-clock timing without normalizing by request count.

    Args:
        total_time: Raw total wall-clock time for the run (not divided by N).
        num_requests: Number of attempted requests (N), including failures.

    Returns:
        A dict with exactly two keys:
        - "unnormalized_e2e": the raw total wall-clock time.
        - "request/s": N / total_time, or 0.0 when total_time <= 0.
    """
    if total_time <= 0:
        return {"unnormalized_e2e": total_time, "request/s": 0.0}

    return {
        "unnormalized_e2e": total_time,
        "request/s": num_requests / total_time,
    }
