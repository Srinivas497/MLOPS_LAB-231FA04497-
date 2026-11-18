"""Generate deterministic synthetic V1 and peak-hour V2 transfer records."""
from pathlib import Path
import csv, random
from .features import engineer_features, write_feature_store

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["record_id", "dataset_version", "expected_train_delay_min",
          "platform_to_bus_stop_distance_m", "passenger_walking_speed_mps",
          "station_crowd_density", "luggage_category", "scheduled_transfer_time_min",
          "lift_escalator_available", "missed_connection"]

def generate(version="v1", n=300, seed=2026):
    rng = random.Random(seed + (1000 if version == "v2" else 0))
    rows=[]
    for i in range(n):
        peak = version == "v2"
        delay = min(35, max(0, rng.gauss(8 if peak else 4, 7 if peak else 4)))
        distance = rng.uniform(80, 650)
        speed = rng.uniform(0.65, 1.55)
        crowd = min(1, max(0, rng.betavariate(4 if peak else 2.3, 2.4 if peak else 3.8)))
        luggage = rng.choices(["None","Light","Heavy"], [0.32,0.48,0.20])[0]
        transfer = rng.uniform(6, 28)
        access = rng.choices(["Yes","No"], [0.78,0.22])[0]
        row={"record_id":f"{version.upper()}-{i+1:04d}","dataset_version":version,
             "expected_train_delay_min":round(delay,2),
             "platform_to_bus_stop_distance_m":round(distance,1),
             "passenger_walking_speed_mps":round(speed,2),
             "station_crowd_density":round(crowd,3),"luggage_category":luggage,
             "scheduled_transfer_time_min":round(transfer,2),"lift_escalator_available":access}
        margin=engineer_features(row)["transfer_time_margin_min"]
        # Near-zero stochastic buffer represents variability in gates, queues and boarding.
        miss = margin < rng.gauss(0.8 if peak else 0.0, 1.7)
        row["missed_connection"]="Yes" if miss else "No"
        rows.append(row)
    return rows

def save(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

def main():
    raw=ROOT/"data"/"raw"; fs=ROOT/"data"/"feature_store"
    for version in ("v1","v2"):
        rows=generate(version)
        save(rows,raw/f"transfers_{version}.csv")
        write_feature_store(rows,fs/f"transfers_{version}_features.csv")
        print(f"{version}: {len(rows)} records, missed={sum(r['missed_connection']=='Yes' for r in rows)}")

if __name__=="__main__": main()
