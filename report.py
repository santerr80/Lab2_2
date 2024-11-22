import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()
experiments = client.search_experiments()
for exp in experiments:
    print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}")

experiment_id = experiments[0].experiment_id

experiment = mlflow.get_experiment(experiment_id)
print(f"Experiment Name: {experiment.name}")
print(f"Artifact Location: {experiment.artifact_location}")

runs = client.search_runs(experiment_ids=[experiment_id])

for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"Parameters: {run.data.params}")
    print(f"Metrics: {run.data.metrics}")
    print("="*40)