"""Compare classifiers and register the recall-oriented champion in MLflow."""
from pathlib import Path
import json, os
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .generate_data import ROOT, generate
from .features import engineer_features, feature_names, write_feature_store

NUM=["expected_train_delay_min","platform_to_bus_stop_distance_m","passenger_walking_speed_mps","station_crowd_density","scheduled_transfer_time_min","estimated_walking_time_min","transfer_time_margin_min"]
CAT=["luggage_category","lift_escalator_available"]
TARGET="missed_connection"

def make_pipeline(model):
    prep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer()),("scale",StandardScaler())]),NUM),
                            ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CAT)])
    return Pipeline([("features",prep),("classifier",model)])

def fit_and_register():
    rows=generate("v1")+generate("v2")
    for v in ("v1","v2"):
        write_feature_store(generate(v), ROOT/"data"/"feature_store"/f"transfers_{v}_features.csv")
    frame=pd.DataFrame([engineer_features(r) for r in rows])
    X=frame[feature_names()]; y=frame[TARGET].map({"No":0,"Yes":1})
    xtr,xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    models={"decision_tree":DecisionTreeClassifier(max_depth=6,min_samples_leaf=4,class_weight="balanced",random_state=42),
            "gradient_boosting":GradientBoostingClassifier(n_estimators=120,max_depth=2,learning_rate=.05,random_state=42)}
    tracking_uri=(ROOT/"mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("railway-feeder-connection-risk")
    results={}
    fitted={}
    for name,model in models.items():
        pipe=make_pipeline(model)
        with mlflow.start_run(run_name=name) as run:
            pipe.fit(xtr,ytr); pred=pipe.predict(xte); prob=pipe.predict_proba(xte)[:,1]
            metrics={"accuracy":accuracy_score(yte,pred),"precision_missed":precision_score(yte,pred,zero_division=0),
                     "recall_missed":recall_score(yte,pred,zero_division=0),"f1_missed":f1_score(yte,pred,zero_division=0),
                     "roc_auc":roc_auc_score(yte,prob),"confusion_matrix":confusion_matrix(yte,pred).tolist()}
            mlflow.log_param("selection_priority","recall_missed, then f1_missed")
            mlflow.log_metrics({k:v for k,v in metrics.items() if k!="confusion_matrix"})
            mlflow.log_dict({"confusion_matrix":metrics["confusion_matrix"]},"confusion_matrix.json")
            mlflow.sklearn.log_model(pipe,name="model",serialization_format="pickle")
            results[name]={"metrics":metrics,"run_id":run.info.run_id}; fitted[name]=pipe
    champion=max(results,key=lambda k:(results[k]["metrics"]["recall_missed"],results[k]["metrics"]["f1_missed"]))
    # Register only the winner after evaluation.
    with mlflow.start_run(run_name=f"registered_champion_{champion}"):
        mlflow.sklearn.log_model(fitted[champion],name="champion",registered_model_name="FeederConnectionRisk",serialization_format="pickle")
    import joblib
    (ROOT/"models").mkdir(exist_ok=True)
    joblib.dump(fitted[champion],ROOT/"models"/"selected_model.joblib")
    results["selected_model"]=champion
    (ROOT/"reports").mkdir(exist_ok=True)
    (ROOT/"reports"/"metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    card={"model":champion,"registered_name":"FeederConnectionRisk","selection":"highest missed-connection recall, then F1",
          "data":"600 synthetic records, V1 and V2 pooled; fixed random stratified 75/25 split","caveat":"Synthetic benchmark only; scores do not establish real-world performance."}
    (ROOT/"reports"/"model_card.json").write_text(json.dumps(card,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2)); return results

if __name__=="__main__": fit_and_register()
