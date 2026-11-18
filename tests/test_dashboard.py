from src.api import dashboard, dashboard_data

def test_dashboard_page_and_summary():
    page=dashboard()
    summary=dashboard_data()
    assert str(page.path).replace("\\","/").endswith("web/index.html")
    assert summary["datasets"]["v1"]["records"]==300
    assert summary["datasets"]["v2"]["records"]==300
    assert summary["registry"]["name"]=="FeederConnectionRisk"
    assert summary["metrics"]["selected_model"] in {"decision_tree","gradient_boosting"}
