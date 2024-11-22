import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Загрузка данных
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.2,
                                                    random_state=42)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("diabetes")

try:
    experiment_id = mlflow.create_experiment(experiment_name)
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Эксперимент 1: n_estimators=100, max_depth=None
with mlflow.start_run(experiment_id=experiment_id) as run1:
    # Обучение модели
    rf1 = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    rf1.fit(X_train, y_train)

    # Оценка модели
    predictions1 = rf1.predict(X_test)
    mse1 = mean_squared_error(y_test, predictions1)
    r2_1 = r2_score(y_test, predictions1)

    # Логирование параметров, метрик и модели
    metrics = {"mse1": mse1, "r2_1": r2_1}
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", None)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rf1, "model")

# Эксперимент 2: n_estimators=200, max_depth=10
with mlflow.start_run(experiment_id=experiment_id) as run2:
    # Обучение модели
    rf2 = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf2.fit(X_train, y_train)

    # Оценка модели
    predictions2 = rf2.predict(X_test)
    mse2 = mean_squared_error(y_test, predictions2)
    r2_2 = r2_score(y_test, predictions2)

    # Логирование параметров, метрик и модели
    metrics = {"mse2": mse2, "r2_2": r2_2}
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(rf2, "model")

# Проверка запусков
runs = mlflow.search_runs(experiment_ids=[experiment_id])
print("Запуски в эксперименте 'diabetes':")
print(runs)