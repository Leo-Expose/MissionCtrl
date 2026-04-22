import requests
import time
import subprocess
import os

print("Starting server...")
server = subprocess.Popen(["python3", "-m", "uvicorn", "server.app:app", "--port", "8000"])
time.sleep(2)

try:
    print("Resetting env...")
    resp = requests.post("http://localhost:8000/reset", json={"difficulty": "easy"})
    resp.raise_for_status()
    print("Reset OK")
    
    print("Taking steps...")
    for i in range(3):
        resp = requests.post("http://localhost:8000/step", json={"action": f"APPROVE(T{i+1:03d})"})
        resp.raise_for_status()
        res = resp.json()
        print(f"Step {i+1}: reward={res['reward']}, done={res['done']}")
        if res['done']:
            break

    print("Posting to /record...")
    payload = {
        "tier": "easy",
        "score": 0.5,
        "steps": 3,
        "history": [],
        "score_breakdown": {},
        "hallucination_stats": {}
    }
    resp = requests.post("http://localhost:8000/record", json=payload)
    resp.raise_for_status()
    print("Record OK")

    print("Checking /results...")
    resp = requests.get("http://localhost:8000/results")
    resp.raise_for_status()
    data = resp.json()
    print(f"Results: {data}")
    assert len(data) == 1
    print("All tests passed!")
finally:
    server.terminate()
