"""HTTP smoke test for a running local FastAPI app (also used in GitHub Actions)."""
import json
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE=os.environ.get("APP_BASE_URL","http://127.0.0.1:8000")

def request(path,method="GET",body=None):
    data=json.dumps(body).encode() if body is not None else None
    req=Request(BASE+path,data=data,method=method,headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=10) as response:
        return response.status,response.headers,response.read()

def main():
    for _ in range(40):
        try:
            status,_,html=request("/")
            break
        except (URLError,TimeoutError):
            time.sleep(.5)
    else:
        raise RuntimeError("FastAPI did not start within 20 seconds")
    assert status==200 and b"Connection risk" in html
    status,_,css=request("/static/app.css")
    assert status==200 and b".sidebar" in css
    status,_,health=request("/health")
    assert status==200 and json.loads(health)["status"]=="ready"
    status,_,summary=request("/api/dashboard")
    dashboard=json.loads(summary)
    assert status==200 and dashboard["datasets"]["v1"]["records"]>=250
    assert dashboard["datasets"]["v2"]["records"]>=250
    for version in ("v1","v2"):
        status,headers,csv=request(f"/api/data/{version}")
        assert status==200 and headers.get_content_type()=="text/csv"
        assert len(csv.decode().splitlines())>=251
    payload={"expected_train_delay_min":2,"platform_to_bus_stop_distance_m":120,
        "passenger_walking_speed_mps":1.2,"station_crowd_density":.15,
        "luggage_category":"Light","scheduled_transfer_time_min":14,
        "lift_escalator_available":"Yes"}
    status,_,prediction=request("/predict","POST",payload)
    result=json.loads(prediction)
    assert status==200 and result["prediction"] in {"Connection Likely","Connection at Risk"}
    assert 0<=result["missed_connection_probability"]<=1
    print("HTTP smoke test passed: dashboard, datasets, health, and prediction.")

if __name__=="__main__": main()
