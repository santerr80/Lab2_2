# Import the necessary libraries
import mlflow
from mlflow.tracking import MlflowClient
import os

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# List all experiments
client = MlflowClient()
experiments = client.search_experiments()
for exp in experiments:
    print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}")

# Get the experiment ID
experiment_id = experiments[0].experiment_id


# Get the experiment
experiment = mlflow.get_experiment(experiment_id)
print(f"Experiment Name: {experiment.name}")
print(f"Artifact Location: {experiment.artifact_location}")

# List all runs
runs = client.search_runs(experiment_ids=[experiment_id])

# Save the report
filename = os.path.join('reports', 'report.txt')
dirname = os.path.dirname(filename)

if not os.path.exists(dirname):
    os.makedirs(dirname)

with open(filename, "w") as f:
    for run in runs:
        print(f"Run ID: {run.info.run_id}", file=f)
        print(f"Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}", file=f)
        print(f"Parameters: {run.data.params}", file=f)
        print(f"Metrics: {run.data.metrics}", file=f)
        print("="*40, file=f)
