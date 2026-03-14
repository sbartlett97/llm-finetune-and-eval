from __future__ import annotations

import logging
import statistics
import time

import httpx

from src.types import LatencyResults

logger = logging.getLogger(__name__)

_SAMPLE_QUESTION = "I have had a headache for 3 days. What should I do?"


class LatencyBenchmark:
    def __init__(self, serving_url: str, warmup_requests: int = 10, benchmark_requests: int = 100):
        self.serving_url = serving_url.rstrip("/")
        self.warmup_requests = warmup_requests
        self.benchmark_requests = benchmark_requests

    def _send_request(self, client: httpx.Client) -> float:
        start = time.perf_counter()
        response = client.post(
            f"{self.serving_url}/generate",
            json={"question": _SAMPLE_QUESTION, "max_new_tokens": 128},
            timeout=60.0,
        )
        response.raise_for_status()
        return (time.perf_counter() - start) * 1000

    def run(self) -> LatencyResults:
        with httpx.Client() as client:
            logger.info("Warming up with %d requests...", self.warmup_requests)
            for _ in range(self.warmup_requests):
                self._send_request(client)

            logger.info("Running %d benchmark requests...", self.benchmark_requests)
            latencies: list[float] = []
            start_total = time.perf_counter()
            for _ in range(self.benchmark_requests):
                latencies.append(self._send_request(client))
            elapsed = time.perf_counter() - start_total

        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies_sorted)
        p95 = latencies_sorted[int(0.95 * len(latencies_sorted))]
        p99 = latencies_sorted[int(0.99 * len(latencies_sorted))]
        throughput = self.benchmark_requests / elapsed

        return LatencyResults(
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            throughput_rps=throughput,
            num_requests=self.benchmark_requests,
        )
