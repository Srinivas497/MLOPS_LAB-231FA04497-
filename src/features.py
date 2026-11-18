"""Shared transfer feature logic and lightweight feature repository."""
from pathlib import Path
import csv

LUGGAGE_PENALTY = {"None": 0.0, "Light": 0.15, "Heavy": 0.30}

def engineer_features(record):
    distance = float(record["platform_to_bus_stop_distance_m"])
    speed = float(record["passenger_walking_speed_mps"])
    crowd = float(record["station_crowd_density"])
    delay = float(record["expected_train_delay_min"])
    transfer = float(record["scheduled_transfer_time_min"])
    luggage = record["luggage_category"]
    access = record["lift_escalator_available"]
    if distance < 0 or speed <= 0 or not 0 <= crowd <= 1 or delay < 0 or transfer < 0:
        raise ValueError("Distance, delay and transfer must be non-negative; speed positive; crowd in [0, 1].")
    if luggage not in LUGGAGE_PENALTY:
        raise ValueError(f"Unsupported luggage category: {luggage}")
    if access not in ("Yes", "No"):
        raise ValueError("lift_escalator_available must be Yes or No")
    walk = (distance / speed / 60.0) * (1.0 + 0.8 * crowd + LUGGAGE_PENALTY[luggage])
    if access == "No":
        walk += 0.75
    out = dict(record)
    out["estimated_walking_time_min"] = round(walk, 4)
    out["transfer_time_margin_min"] = round(transfer - delay - walk, 4)
    return out

def feature_names():
    return ["expected_train_delay_min", "platform_to_bus_stop_distance_m",
            "passenger_walking_speed_mps", "station_crowd_density",
            "luggage_category", "scheduled_transfer_time_min",
            "lift_escalator_available", "estimated_walking_time_min",
            "transfer_time_margin_min"]

def write_feature_store(records, output_path):
    rows = [engineer_features(r) for r in records]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
