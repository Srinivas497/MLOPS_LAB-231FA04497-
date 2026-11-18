from src.generate_data import generate

def test_generates_versioned_records_with_peak_hour_shift():
    baseline=generate("v1",n=300)
    peak=generate("v2",n=300)
    assert len(baseline)>=250 and len(peak)>=250
    assert all(row["dataset_version"]=="v1" for row in baseline)
    assert all(row["dataset_version"]=="v2" for row in peak)
    baseline_delay=sum(row["expected_train_delay_min"] for row in baseline)/len(baseline)
    peak_delay=sum(row["expected_train_delay_min"] for row in peak)/len(peak)
    baseline_crowd=sum(row["station_crowd_density"] for row in baseline)/len(baseline)
    peak_crowd=sum(row["station_crowd_density"] for row in peak)/len(peak)
    assert peak_delay>baseline_delay
    assert peak_crowd>baseline_crowd
