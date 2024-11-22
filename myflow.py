import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Загрузка данных
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.2,
                                                    random_state=42)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("diabetes")


# Начало эксперимента
with mlflow.start_run():
    # Обучение модели
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Оценка модели
    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Логирование параметров, метрик и модели
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(rf, "model")
mlflow.end_run()
