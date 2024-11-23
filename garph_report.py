import mlflow
import mlflow.tracking
import matplotlib.pyplot as plt
import pandas as pd

# Setup mlflow
mlflow.set_tracking_uri("http://localhost:5000")

# Get the experiment ID
experiment_name = "diabetes"
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Get the runs
runs = mlflow.search_runs(experiment_ids=[experiment_id])

pd.set_option('display.max_columns', None)
print(runs.head())

# Get the metrics r2
metrics = runs[["tags.mlflow.runName", "metrics.r2"]]

# Plot the metrics
plt.figure(figsize=(10, 6))
plt.plot(metrics["tags.mlflow.runName"], metrics["metrics.r2"], marker='o')
plt.xlabel('Run ID')
plt.ylabel('Metric Value')
plt.title(f'Metric R2 Overview for {experiment_name}')
plt.grid(True)
plt.savefig("reports/metrics_r2.png")
plt.close()


# Get the metrics mse
metrics = runs[["tags.mlflow.runName", "metrics.mse"]]

# Plot the metrics
plt.figure(figsize=(10, 6))
plt.plot(metrics["tags.mlflow.runName"], metrics["metrics.mse"], marker='o')
plt.xlabel('Run ID')
plt.ylabel('Metric Value')
plt.title(f'Metric MSE Overview for {experiment_name}')
plt.grid(True)
plt.savefig("reports/metrics_mse.png")
plt.close()