
import time
import csv
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from gradio_client import Client

SERVER_URL = "http://127.0.0.1:7860"
IMAGE_PATH = str(
    Path("data/processed/test/COVID/COVID-1018.png").resolve()
)

TOTAL_REQUESTS = 50
CONCURRENT_USERS = 50

thread_local = threading.local()


def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = Client(SERVER_URL, verbose=False)
    return thread_local.client


def send_request(request_id):
    start = time.perf_counter()

    try:
        client = get_client()
        result = client.predict(
            IMAGE_PATH,
            api_name="/predict_xray"
        )

        elapsed = time.perf_counter() - start

        return {
            "request": request_id,
            "status": "SUCCESS",
            "latency_seconds": round(elapsed, 3),
            "result": str(result)
        }

    except Exception as error:
        elapsed = time.perf_counter() - start

        return {
            "request": request_id,
            "status": "FAILED",
            "latency_seconds": round(elapsed, 3),
            "result": str(error)
        }


if __name__ == "__main__":
    print("MedScan 50-request load test")
    print("Image:", IMAGE_PATH)
    print("Concurrent requests:", CONCURRENT_USERS)

    start_time = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
        futures = [
            executor.submit(send_request, i + 1)
            for i in range(TOTAL_REQUESTS)
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Request {result['request']}: "
                f"{result['status']} - "
                f"{result['latency_seconds']} seconds"
            )

    total_time = time.perf_counter() - start_time

    successful = [
        r for r in results if r["status"] == "SUCCESS"
    ]

    failed = TOTAL_REQUESTS - len(successful)

    latencies = sorted(
        r["latency_seconds"] for r in successful
    )

    average_latency = (
        sum(latencies) / len(latencies) if latencies else 0
    )

    p95_latency = (
        latencies[min(len(latencies) - 1,
                      int(0.95 * len(latencies)))]
        if latencies else 0
    )

    throughput = len(successful) / total_time

    with open("results/load_test_results.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "request",
                "status",
                "latency_seconds",
                "result"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== LOAD TEST SUMMARY =====")
    print("Total requests:", TOTAL_REQUESTS)
    print("Successful:", len(successful))
    print("Failed:", failed)
    print("Total time:", round(total_time, 2), "seconds")
    print("Average latency:", round(average_latency, 3), "seconds")
    print("P95 latency:", round(p95_latency, 3), "seconds")
    print("Throughput:", round(throughput, 2), "requests/second")
    print("Results saved to results/load_test_results.csv")
