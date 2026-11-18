from src.features import engineer_features

def test_walking_time_and_margin():
    x={"platform_to_bus_stop_distance_m":120,"passenger_walking_speed_mps":1.2,
       "station_crowd_density":0,"expected_train_delay_min":2,"scheduled_transfer_time_min":8,
       "luggage_category":"None","lift_escalator_available":"Yes"}
    y=engineer_features(x)
    assert y["estimated_walking_time_min"]==1.6667
    assert y["transfer_time_margin_min"]==4.3333

def test_congestion_and_access_add_walking_time():
    base={"platform_to_bus_stop_distance_m":300,"passenger_walking_speed_mps":1,
          "station_crowd_density":0,"expected_train_delay_min":0,"scheduled_transfer_time_min":10,
          "luggage_category":"None","lift_escalator_available":"Yes"}
    busy={**base,"station_crowd_density":1,"lift_escalator_available":"No"}
    assert engineer_features(busy)["estimated_walking_time_min"]>engineer_features(base)["estimated_walking_time_min"]
