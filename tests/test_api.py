from src.api import predict, TransferRequest

def test_prediction_function_with_fitted_model():
    body=predict(TransferRequest(expected_train_delay_min=2,platform_to_bus_stop_distance_m=120,
      passenger_walking_speed_mps=1.2,station_crowd_density=0.15,luggage_category="Light",
      scheduled_transfer_time_min=14,lift_escalator_available="Yes"))
    assert body["prediction"] in {"Connection Likely","Connection at Risk"}
    assert 0<=body["missed_connection_probability"]<=1
