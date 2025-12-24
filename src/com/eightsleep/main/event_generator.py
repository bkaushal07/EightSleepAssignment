# event_generator.py
import time
import json
import random
import requests
from concurrent.futures import ThreadPoolExecutor

def make_event(uid):
    ts = int(time.time())
    features = [random.random(), random.random(), random.random()]
    return {"user_id": f"user-{uid}", "timestamp": ts, "features": features}

def post_batch(url, batch):
    try:
        r = requests.post(url, json={"events": batch}, timeout=5)
        return r.status_code
    except Exception as e:
        print("post error", e)
    return None

def run(target_url="http://server:8000/ingest", rps=5000, duration_sec=60, users=10000):
    batch_size = 10
    batches_per_sec = rps // batch_size
    if batches_per_sec <= 0:
        raise ValueError("RPS must be at least equal to the batch size (10).")
    interval = 1.0 / batches_per_sec # seconds between batches
    end = time.time() + duration_sec
    with ThreadPoolExecutor(max_workers=50) as ex:
        next_send = time.time()
        sent = 0
        while time.time() < end:
        # Build batch
            batch = [make_event(random.randint(0, users - 1)) for _ in
            range(batch_size)]
            ex.submit(post_batch, target_url, batch)
            sent += batch_size
            # Sleep until next interval
            next_send += interval
            sleep_time = next_send - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
    print(f"Sent approximately {sent} events at target RPS={rps}")

if __name__ == "__main__":
    run()