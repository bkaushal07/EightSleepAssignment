# server.py
import time
import torch
import threading
from collections import defaultdict, deque
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import numpy as np
import traceback
from create_model import InefficientModel

model = torch.load("inefficient_model.pt", map_location="cpu", weights_only=False)
if not isinstance(model, InefficientModel):
    m = InefficientModel()
    try:
        m.load_state_dict(torch.load("inefficient_model.pt", map_location="cpu"))
        model = m
    except Exception:
        model = m

model.eval()

user_scores = defaultdict(lambda: deque(maxlen=300))  # store last 5 mins (assuming 1 event/sec)
user_timestamps = defaultdict(lambda: deque(maxlen=300))
lock = threading.Lock()

stats = {"requests": 0, "ingested_events": 0, "start_time": time.time()}

app = FastAPI()

class Event(BaseModel):
    user_id: str
    timestamp: int
    features: list

class EventBatch(BaseModel):
    events: list[Event]

@app.post("/ingest")
def ingest_events(batch: EventBatch):
    stats["requests"] += 1
    results = []
    with torch.no_grad():
        for e in batch.events:
            x = torch.tensor(e.features, dtype=torch.float32)
            score = model(x).item()
            with lock:
                user_scores[e.user_id].append((e.timestamp, score))
                stats["ingested_events"] += 1
            results.append({
                "user_id": e.user_id,
                "timestamp": e.timestamp,
                "features": e.features,
                "predicted_score": score
            })
    return {"status": "ok", "count": len(batch.events), "results": results}

@app.get("/users/{user_id}/median")
def get_user_median(user_id: str):
    try:
        with lock:
            recent_scores = [score for (timestamp, score) in user_scores[user_id] if time.time() - timestamp <= 300]
            if not recent_scores:
                return {"user_id": user_id, "median": None}
            return {"user_id": user_id, "median": float(np.median(recent_scores))}
    except Exception as e:
        print(f"Error in /users/{user_id}/median:", e)
        traceback.print_exc()
        return {"error": str(e)}

@app.get("/stats")
def get_stats():
    try:
        with lock:
            total_scores = sum(len(v) for v in user_scores.values())
            all_medians = []
            for v in user_scores.values():
                recent_scores = [score for (timestamp, score) in v if time.time() - timestamp <= 300]
                if recent_scores:
                    all_medians.append(np.median(recent_scores))
            overall_median = float(np.median(all_medians)) if all_medians else None

            uptime = time.time() - stats["start_time"]
            avg_rps = stats["requests"] / uptime if uptime > 0 else 0

            return {
                "requests": stats["requests"],
                "ingested_events": stats["ingested_events"],
                "users_tracked": len(user_scores),
                "avg_rps": round(avg_rps, 2),
                "total_scores_stored": total_scores,
                "overall_median_score": overall_median,
            }
    except Exception as e:
        print("Error in /stats endpoint:", e)
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)