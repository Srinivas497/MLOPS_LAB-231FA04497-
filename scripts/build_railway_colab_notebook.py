"""Create a self-contained Colab notebook for the railway-feeder MLOps project."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Railway_Feeder_Connection_Risk_MLOps_Colab.ipynb"

cells = [
("markdown", r'''# Predicting Missed Connections Between Railway Trains and Feeder Buses

**End-to-end MLOps project | Google Colab notebook**

This notebook builds the railway connection-risk project step by step. It creates two versioned synthetic datasets, engineers station-transfer features, compares Decision Tree and Gradient Boosting classifiers, registers the recall-first champion with MLflow, and serves/test predictions through FastAPI.

**Prediction target:** `missed_connection` (`Yes` / `No`). The records and labels are synthetic, so the scores are for demonstrating a reproducible workflow, not for real passenger dispatch decisions.

### Notebook map

1. Set up the runtime and project folders.
2. Generate V1 baseline and V2 peak-hour transfer records.
3. Build reusable walking-time and transfer-margin features.
4. Train and compare two classifiers with a held-out test set.
5. Track runs and register the champion in MLflow.
6. Build and exercise the FastAPI prediction service.
7. Review CI/CD and the project outputs.'''),
("code", r'''# 1. Install the notebook dependencies (run this cell first in Colab)
%pip -q install numpy pandas scikit-learn matplotlib joblib mlflow fastapi uvicorn pydantic pytest'''),
("code", r'''# 2. Imports and project folders
import json, os, sys, time, threading, warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

SEED = 2026
os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE','true')
ROOT = Path('/content/railway-feeder-connection-risk') if Path('/content').exists() else Path.cwd() / 'railway-feeder-connection-risk'
for folder in ['data/raw','data/feature_store','models','reports','mlruns']:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
print('Project folder:', ROOT)
print('Runtime:', sys.version.split()[0])'''),
("markdown", r'''## Task 1 - Versioned synthetic transfer data

Each version contains 300 records. V1 represents regular conditions; V2 increases expected train delays and station crowd density to represent peak-hour congestion. Target labels include a small random buffer for unobserved queuing and boarding variability.'''),
("code", r'''# 3. Generate and save V1 and V2 datasets
RAW_COLUMNS = [
    'record_id','dataset_version','expected_train_delay_min',
    'platform_to_bus_stop_distance_m','passenger_walking_speed_mps',
    'station_crowd_density','luggage_category','scheduled_transfer_time_min',
    'lift_escalator_available','missed_connection'
]

def generate_dataset(version, n=300, seed=SEED):
    """Generate reproducible synthetic train-to-feeder-bus transfer records."""
    if version not in {'v1','v2'}:
        raise ValueError("version must be 'v1' or 'v2'")
    peak = version == 'v2'
    rng = np.random.default_rng(seed + (1000 if peak else 0))
    delay = np.clip(rng.normal(8 if peak else 4, 7 if peak else 4, n), 0, 35)
    distance = rng.uniform(80, 650, n)
    speed = rng.uniform(.65, 1.55, n)
    crowd = np.clip(rng.beta(4 if peak else 2.3, 2.4 if peak else 3.8, n), 0, 1)
    luggage = rng.choice(['None','Light','Heavy'], n, p=[.32,.48,.20])
    transfer = rng.uniform(6, 28, n)
    access = rng.choice(['Yes','No'], n, p=[.78,.22])
    luggage_penalty = pd.Series(luggage).map({'None':0.0,'Light':.15,'Heavy':.30}).to_numpy()
    walk = (distance / speed / 60.0) * (1 + .8*crowd + luggage_penalty) + np.where(access == 'No', .75, 0)
    margin = transfer - delay - walk
    # A noisy threshold represents queues, exit locations and boarding variation.
    missed = margin < rng.normal(.8 if peak else 0, 1.7, n)
    start = datetime(2026, 2 if not peak else 6, 1, tzinfo=timezone.utc)
    return pd.DataFrame({
        'record_id':[f'{version.upper()}-{i+1:04d}' for i in range(n)],
        'dataset_version':version,
        'expected_train_delay_min':np.round(delay,2),
        'platform_to_bus_stop_distance_m':np.round(distance,1),
        'passenger_walking_speed_mps':np.round(speed,2),
        'station_crowd_density':np.round(crowd,3),
        'luggage_category':luggage,
        'scheduled_transfer_time_min':np.round(transfer,2),
        'lift_escalator_available':access,
        'missed_connection':np.where(missed,'Yes','No'),
        'timestamp':[(start+timedelta(minutes=10*i)).isoformat() for i in range(n)]
    })

raw_v1 = generate_dataset('v1')
raw_v2 = generate_dataset('v2')
manifest = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'versions':{}}
for version, frame, description in [
    ('v1',raw_v1,'Baseline service with moderate delays and crowd density'),
    ('v2',raw_v2,'Peak-hour congestion with increased train delays and crowd density')]:
    path = ROOT / 'data' / 'raw' / f'transfers_{version}.csv'
    frame.to_csv(path,index=False)
    manifest['versions'][version] = {'description':description,'path':str(path.relative_to(ROOT)),
        'rows':len(frame),'missed_connections':int((frame.missed_connection=='Yes').sum()),
        'missed_rate':float((frame.missed_connection=='Yes').mean())}
(ROOT/'data'/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
display(pd.DataFrame(manifest['versions']).T)
assert len(raw_v1) >= 250 and len(raw_v2) >= 250
assert raw_v2.expected_train_delay_min.mean() > raw_v1.expected_train_delay_min.mean()
assert raw_v2.station_crowd_density.mean() > raw_v1.station_crowd_density.mean()
print('Both dataset versions created and peak-hour shift checks passed.')'''),
("code", r'''# 4. Compare the two operating conditions
fig, axes = plt.subplots(1, 2, figsize=(11,4))
axes[0].hist(raw_v1.expected_train_delay_min,bins=18,alpha=.7,label='V1 baseline')
axes[0].hist(raw_v2.expected_train_delay_min,bins=18,alpha=.7,label='V2 peak hour')
axes[0].set(title='Expected train delay',xlabel='Minutes',ylabel='Records'); axes[0].legend(); axes[0].grid(alpha=.2)
axes[1].hist(raw_v1.station_crowd_density,bins=18,alpha=.7,label='V1 baseline')
axes[1].hist(raw_v2.station_crowd_density,bins=18,alpha=.7,label='V2 peak hour')
axes[1].set(title='Station crowd density',xlabel='Density (0 to 1)',ylabel='Records'); axes[1].legend(); axes[1].grid(alpha=.2)
plt.tight_layout(); plt.show()
pd.DataFrame([{'version':'V1','records':len(raw_v1),'missed_rate':raw_v1.missed_connection.eq('Yes').mean(),
               'mean_delay_min':raw_v1.expected_train_delay_min.mean(),'mean_crowd':raw_v1.station_crowd_density.mean()},
              {'version':'V2 peak hour','records':len(raw_v2),'missed_rate':raw_v2.missed_connection.eq('Yes').mean(),
               'mean_delay_min':raw_v2.expected_train_delay_min.mean(),'mean_crowd':raw_v2.station_crowd_density.mean()}]).round(3)'''),
("markdown", r'''## Task 2 - Automated preprocessing and reusable feature store

Estimated walking time accounts for distance, walking speed, crowding, luggage and whether lift/escalator access is available. Transfer-time margin is the scheduled connection window after subtracting expected train delay and walking time. The same function is used for training features and API predictions.'''),
("code", r'''# 5. Validate, engineer and persist reusable features
LUGGAGE_PENALTY = {'None':0.0,'Light':0.15,'Heavy':0.30}
NUMERIC_FEATURES = ['expected_train_delay_min','platform_to_bus_stop_distance_m',
    'passenger_walking_speed_mps','station_crowd_density','scheduled_transfer_time_min',
    'estimated_walking_time_min','transfer_time_margin_min']
CATEGORICAL_FEATURES = ['luggage_category','lift_escalator_available']
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = 'missed_connection'

def engineer_features(frame):
    out = frame.copy()
    if (out.platform_to_bus_stop_distance_m < 0).any(): raise ValueError('Distance must be non-negative')
    if (out.passenger_walking_speed_mps <= 0).any(): raise ValueError('Walking speed must be positive')
    if not out.station_crowd_density.between(0,1).all(): raise ValueError('Crowd density must be in [0,1]')
    if not out.luggage_category.isin(LUGGAGE_PENALTY).all(): raise ValueError('Invalid luggage category')
    if not out.lift_escalator_available.isin(['Yes','No']).all(): raise ValueError('Access field must be Yes or No')
    penalty = out.luggage_category.map(LUGGAGE_PENALTY)
    out['estimated_walking_time_min'] = (
        out.platform_to_bus_stop_distance_m / out.passenger_walking_speed_mps / 60.0
        * (1 + .8*out.station_crowd_density + penalty)
        + np.where(out.lift_escalator_available.eq('No'),.75,0)
    )
    out['transfer_time_margin_min'] = (
        out.scheduled_transfer_time_min - out.expected_train_delay_min - out.estimated_walking_time_min
    )
    return out

feature_frames = {}
feature_registry = {'versions':{},'features':['estimated_walking_time_min','transfer_time_margin_min']}
for version, raw in [('v1',raw_v1),('v2',raw_v2)]:
    features = engineer_features(raw)
    feature_frames[version] = features
    feature_path = ROOT/'data'/'feature_store'/f'transfers_{version}_features.csv'
    features.to_csv(feature_path,index=False)
    feature_registry['versions'][version] = {'path':str(feature_path.relative_to(ROOT)),'rows':len(features)}
(ROOT/'data'/'feature_store'/'registry.json').write_text(json.dumps(feature_registry,indent=2),encoding='utf-8')
all_data = pd.concat(feature_frames.values(),ignore_index=True)
display(all_data[MODEL_FEATURES+[TARGET]].head())
print('Feature-store records:',len(all_data))'''),
("code", r'''# 6. Unit-check the transfer calculation
unit_example = pd.DataFrame([{
    'platform_to_bus_stop_distance_m':120,'passenger_walking_speed_mps':1.2,
    'station_crowd_density':0.0,'luggage_category':'None','lift_escalator_available':'Yes',
    'scheduled_transfer_time_min':8,'expected_train_delay_min':2
}])
unit_result = engineer_features(unit_example).iloc[0]
assert round(unit_result.estimated_walking_time_min,4) == 1.6667
assert round(unit_result.transfer_time_margin_min,4) == 4.3333
busy_example = unit_example.copy()
busy_example.loc[0,['station_crowd_density','lift_escalator_available','luggage_category']] = [1.0,'No','Heavy']
assert engineer_features(busy_example).estimated_walking_time_min.iloc[0] > unit_result.estimated_walking_time_min
print('PASS: walking-time calculation, transfer margin, crowd, access and luggage checks.')'''),
("markdown", r'''## Task 3 - Train and compare classifiers

A stratified test set is held out until evaluation. The remaining data is split into training and validation sets. Model selection uses validation missed-connection recall, followed by F1 as a tie-breaker. The test set reports final performance for both classifiers.'''),
("code", r'''# 7. Fit Decision Tree and Gradient Boosting candidates
all_data[TARGET+'_binary'] = all_data[TARGET].map({'No':0,'Yes':1})
strata = all_data.dataset_version.astype(str) + '_' + all_data[TARGET].astype(str)
train_val, test_df = train_test_split(all_data,test_size=.25,random_state=SEED,stratify=strata)
train_df, val_df = train_test_split(train_val,test_size=.20,random_state=SEED,
    stratify=train_val.dataset_version.astype(str)+'_'+train_val[TARGET].astype(str))
X_train, y_train = train_df[MODEL_FEATURES], train_df[TARGET+'_binary']
X_val, y_val = val_df[MODEL_FEATURES], val_df[TARGET+'_binary']
X_train_val, y_train_val = train_val[MODEL_FEATURES], train_val[TARGET+'_binary']
X_test, y_test = test_df[MODEL_FEATURES], test_df[TARGET+'_binary']

def build_pipeline(classifier):
    numeric = Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categorical = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
                            ('onehot',OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('numeric',numeric,NUMERIC_FEATURES),('categorical',categorical,CATEGORICAL_FEATURES)])
    return Pipeline([('preprocessor',prep),('classifier',classifier)])

candidates = {
    'decision_tree':DecisionTreeClassifier(max_depth=6,min_samples_leaf=4,class_weight='balanced',random_state=SEED),
    'gradient_boosting':GradientBoostingClassifier(n_estimators=120,max_depth=2,learning_rate=.05,random_state=SEED)
}
validation_models, validation_metrics = {}, {}
for name, estimator in candidates.items():
    pipeline = build_pipeline(estimator).fit(X_train,y_train)
    pred = pipeline.predict(X_val)
    validation_models[name] = pipeline
    validation_metrics[name] = {
        'recall_missed':recall_score(y_val,pred,zero_division=0),
        'precision_missed':precision_score(y_val,pred,zero_division=0),
        'f1_missed':f1_score(y_val,pred,zero_division=0)
    }
best_name = max(validation_metrics,key=lambda name:(validation_metrics[name]['recall_missed'],validation_metrics[name]['f1_missed']))
print('Validation results (selection split):')
display(pd.DataFrame(validation_metrics).T.round(3))
print('Selected champion:',best_name)

# Refit both pipelines on train + validation, then score the untouched test set once.
trained, test_metrics = {}, {}
for name, estimator in candidates.items():
    pipeline = build_pipeline(estimator).fit(X_train_val,y_train_val)
    prediction = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:,1]
    matrix = confusion_matrix(y_test,prediction,labels=[0,1])
    test_metrics[name] = {
        'accuracy':accuracy_score(y_test,prediction),
        'precision_missed':precision_score(y_test,prediction,zero_division=0),
        'recall_missed':recall_score(y_test,prediction,zero_division=0),
        'f1_missed':f1_score(y_test,prediction,zero_division=0),
        'roc_auc':roc_auc_score(y_test,probability),
        'confusion_matrix':matrix.tolist()
    }
    trained[name] = pipeline
best_model = trained[best_name]
metrics_table = pd.DataFrame(test_metrics).T
display(metrics_table.round(3))
print('Selected by validation recall; test recall is reported independently.')'''),
("code", r'''# 8. Visualize classifier comparison and champion confusion matrix
metrics_table[['accuracy','precision_missed','recall_missed','f1_missed']].plot(
    kind='bar',figsize=(9,4),ylim=(0,1),title='Classifier performance on held-out test set')
plt.ylabel('Score'); plt.xticks(rotation=0); plt.grid(axis='y',alpha=.25); plt.show()
champion_matrix = np.array(test_metrics[best_name]['confusion_matrix'])
fig, ax = plt.subplots(figsize=(4.5,4))
im = ax.imshow(champion_matrix,cmap='Blues')
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No miss','Missed'],yticklabels=['No miss','Missed'],
       xlabel='Predicted',ylabel='Actual',title=f'{best_name.replace("_"," ").title()} confusion matrix')
for (i,j),value in np.ndenumerate(champion_matrix): ax.text(j,i,str(value),ha='center',va='center',color='black')
fig.colorbar(im,ax=ax); plt.tight_layout(); plt.show()'''),
("code", r'''# 9. Save the selected pipeline and evaluation artifacts
model_path = ROOT/'models'/'selected_model.joblib'
joblib.dump(best_model,model_path)
summary = {
    'selected_model':best_name,
    'selection_rule':'highest validation missed-connection recall, then validation F1',
    'records_total':len(all_data),'train_records':len(train_df),'validation_records':len(val_df),'test_records':len(test_df),
    'positive_class':'missed_connection = Yes',
    'validation_metrics':validation_metrics,'test_metrics':test_metrics,
    'limitation':'Synthetic data and synthetic target; scores are not real-world performance.'
}
(ROOT/'reports'/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(ROOT/'models'/'model_card.json').write_text(json.dumps({
    'model':best_name,'features':MODEL_FEATURES,'registry_name':'FeederConnectionRisk',
    'selection_rule':summary['selection_rule'],'data':'600 synthetic V1/V2 transfers'
},indent=2),encoding='utf-8')
print('Saved selected model:',model_path)
print('Saved metrics:',ROOT/'reports'/'metrics.json')'''),
("markdown", r'''## Task 4 - MLflow experiment tracking and model registry

Both candidates are tracked. The model selected on validation recall is registered as `FeederConnectionRisk`. The notebook uses a local MLflow filesystem store so experiment runs and the model registry remain together in the project folder.'''),
("code", r'''# 10. Track both experiments and register the champion
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow_db = ROOT/'mlflow.db'
tracking_uri = (ROOT/'mlruns').resolve().as_uri()
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)
mlflow.set_experiment('railway-feeder-connection-risk')
for name,pipeline in trained.items():
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param('model_type',name)
        mlflow.log_param('selection_rule',summary['selection_rule'])
        mlflow.log_param('data_versions','v1,v2')
        mlflow.log_metrics({key:float(value) for key,value in test_metrics[name].items() if key!='confusion_matrix'})
        mlflow.log_dict({'confusion_matrix':test_metrics[name]['confusion_matrix']},'confusion_matrix.json')
        mlflow.log_artifact(str(ROOT/'data'/'manifest.json'))
        mlflow.sklearn.log_model(pipeline,name='model',serialization_format='pickle',
            registered_model_name='FeederConnectionRisk' if name==best_name else None,
            signature=infer_signature(X_train_val,pipeline.predict(X_train_val)),
            input_example=X_train_val.head(3))
        print(name,'run:',run.info.run_id)

client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='FeederConnectionRisk'")
latest_version = max(int(v.version) for v in versions)
print('Registered model: FeederConnectionRisk')
print('Registered version:',latest_version)
print('MLflow tracking URI:',tracking_uri)'''),
("markdown", r'''## Task 5 - FastAPI serving and prediction checks

The API and training pipeline share the same transfer feature function. `/predict` accepts the original transfer conditions and returns one of the required passenger-facing labels, probability, estimated walking time and transfer-time margin.'''),
("code", r'''# 11. Define and test the FastAPI prediction service
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(title='Railway Feeder Connection Risk',version='1.0.0')
class TransferRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_train_delay_min: float = Field(ge=0)
    platform_to_bus_stop_distance_m: float = Field(ge=0)
    passenger_walking_speed_mps: float = Field(gt=0)
    station_crowd_density: float = Field(ge=0,le=1)
    luggage_category: Literal['None','Light','Heavy']
    scheduled_transfer_time_min: float = Field(ge=0)
    lift_escalator_available: Literal['Yes','No']

@app.get('/health')
def health():
    return {'status':'ready','model':best_name,'registry':'FeederConnectionRisk','version':latest_version}

@app.post('/predict')
def predict(request: TransferRequest):
    raw = pd.DataFrame([request.model_dump()])
    features = engineer_features(raw)
    probability = float(best_model.predict_proba(features[MODEL_FEATURES])[0,1])
    label = 'Connection at Risk' if probability >= .5 else 'Connection Likely'
    return {'prediction':label,'missed_connection_probability':round(probability,4),
        'estimated_walking_time_min':round(float(features.estimated_walking_time_min.iloc[0]),4),
        'transfer_time_margin_min':round(float(features.transfer_time_margin_min.iloc[0]),4)}

sample_request = {
    'expected_train_delay_min':2,'platform_to_bus_stop_distance_m':120,
    'passenger_walking_speed_mps':1.2,'station_crowd_density':.15,
    'luggage_category':'Light','scheduled_transfer_time_min':14,
    'lift_escalator_available':'Yes'
}
sample_result = predict(TransferRequest(**sample_request))
assert sample_result['prediction'] in {'Connection Likely','Connection at Risk'}
assert 0 <= sample_result['missed_connection_probability'] <= 1
assert sample_result['estimated_walking_time_min'] >= 0
print('Sample prediction:',sample_result)

high_risk_request = TransferRequest(expected_train_delay_min=30,platform_to_bus_stop_distance_m=600,
    passenger_walking_speed_mps=.7,station_crowd_density=1,luggage_category='Heavy',
    scheduled_transfer_time_min=6,lift_escalator_available='No')
high_risk_result = predict(high_risk_request)
assert high_risk_result['prediction']=='Connection at Risk'
assert high_risk_result['transfer_time_margin_min'] < 0
print('High-risk scenario:',high_risk_result)
print('PASS: prediction labels, probability bounds and transfer calculations.')'''),
("code", r'''# 12. Start FastAPI and test its real HTTP routes
import urllib.request
import socket
from urllib.error import URLError, HTTPError
import uvicorn

port_socket = socket.socket()
port_socket.bind(('127.0.0.1',0))
api_port = port_socket.getsockname()[1]
port_socket.close()

def serve_api():
    uvicorn.run(app,host='0.0.0.0',port=api_port,log_level='warning')

server_thread = globals().get('server_thread')
if server_thread is None or not server_thread.is_alive():
    server_thread = threading.Thread(target=serve_api,daemon=True)
    server_thread.start()

for attempt in range(30):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{api_port}/health',timeout=2) as response:
            health_result = json.loads(response.read())
        break
    except (URLError,TimeoutError):
        time.sleep(.5)
else:
    raise RuntimeError(f'FastAPI did not start on port {api_port}')

payload = json.dumps(sample_request).encode()
request = urllib.request.Request(f'http://127.0.0.1:{api_port}/predict',data=payload,
    headers={'Content-Type':'application/json'},method='POST')
with urllib.request.urlopen(request,timeout=10) as response:
    prediction_result = json.loads(response.read())
bad_payload = json.dumps({**sample_request,'station_crowd_density':1.5}).encode()
bad_request = urllib.request.Request(f'http://127.0.0.1:{api_port}/predict',data=bad_payload,
    headers={'Content-Type':'application/json'},method='POST')
try:
    urllib.request.urlopen(bad_request,timeout=10)
    invalid_status = 200
except HTTPError as error:
    invalid_status = error.code
assert health_result['status']=='ready'
assert prediction_result['prediction'] in {'Connection Likely','Connection at Risk'}
assert invalid_status==422
print('GET /health:',health_result)
print('POST /predict:',prediction_result)
print('Invalid crowd density rejected with HTTP',invalid_status)
print('3/3 FastAPI HTTP checks passed.')'''),
("code", r'''# 13. Start the interactive service in Colab
# The API process was started and checked in the previous cell. This cell
# displays the live service inside Colab, with Swagger at the /docs route.
from IPython.display import display, HTML
try:
    from google.colab.output import serve_kernel_port_as_iframe
    display(HTML(serve_kernel_port_as_iframe(api_port)))
    print('API is available in the embedded preview; Swagger UI is at /docs.')
except ImportError:
    print(f'API process started on port {api_port}. Open http://127.0.0.1:{api_port}/docs in a local notebook environment.')'''),
("markdown", r'''## Task 6 - CI/CD and final project outputs

The project repository contains `.github/workflows/ci.yml`; on pushes and pull requests it installs dependencies, regenerates V1/V2, trains/registers the selected model, runs unit tests, then starts FastAPI for HTTP smoke tests. In this notebook, the matching feature-calculation assertions and FastAPI HTTP checks were executed above.

All scores are based on synthetic records. For real operations, replace the generator and synthetic target with observed arrival, station-walk, bus departure and passenger outcome data, then validate by date and station before using predictions to make service decisions.'''),
("code", r'''# 14. Final run summary
final_summary = {
    'project_folder':str(ROOT),
    'dataset_records':{v:len(frame) for v,frame in [('v1',raw_v1),('v2',raw_v2)]},
    'feature_store_versions':list(feature_frames),
    'models_compared':list(test_metrics),
    'selected_model':best_name,
    'registered_model':'FeederConnectionRisk',
    'registered_version':latest_version,
    'sample_prediction':sample_result,
    'api_http_checks':'3/3 passed'
}
print(json.dumps(final_summary,indent=2))
print('Generated files:')
for path in [ROOT/'data'/'manifest.json',ROOT/'data'/'feature_store'/'registry.json',
             ROOT/'reports'/'metrics.json',ROOT/'models'/'selected_model.joblib',ROOT/'mlruns']:
    print(('OK' if path.exists() else 'MISSING'),path.relative_to(ROOT))'''),
("markdown", r'''## Conclusion

This notebook demonstrates dataset versioning, reusable feature engineering, recall-focused classifier comparison, MLflow model registration, FastAPI serving and automated checks for the railway-to-feeder-bus connection-risk project.

**Important limitation:** Both features and target labels are synthetic. The measured performance does not establish passenger-level real-world accuracy.''')
]

def make_cell(cell_type, source):
    source = source.strip("\n") + "\n"
    base = {"cell_type":cell_type,"metadata":{},"source":source.splitlines(keepends=True)}
    if cell_type == "code":
        base.update({"execution_count":None,"outputs":[]})
    return base

notebook = {
    "cells":[make_cell(kind,source) for kind,source in cells],
    "metadata":{
        "colab":{"name":OUT.name,"provenance":[]},
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3"}
    },
    "nbformat":4,
    "nbformat_minor":5
}
OUT.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
print(f"Created {OUT} with {len(cells)} cells.")
