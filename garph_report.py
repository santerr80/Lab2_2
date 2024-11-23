import mlflow
import mlflow.tracking
import matplotlib.pyplot as plt
import pandas as pd

# Подключитесь к вашему MLflow серверу (если используете удаленный сервер)
mlflow.set_tracking_uri("http://localhost:5000")

# Получите данные метрик для всех запусков эксперимента
experiment_name = "diabetes"
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

runs = mlflow.search_runs(experiment_ids=[experiment_id])

pd.set_option('display.max_columns', None)

print(runs.head())

# Извлеките метрики из запусков
metrics = runs[["tags.mlflow.runName", "metrics.mse"]]

# Постройте график
plt.figure(figsize=(10, 6))
plt.plot(metrics["tags.mlflow.runName"], metrics["metrics.mse"], marker='o')
plt.xlabel('Run ID')
plt.ylabel('Metric Value')
plt.title(f'Metric Overview for {experiment_name}')
plt.grid(True)
plt.show()
