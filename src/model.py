import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier, Pool

def train_model(X_train, y_train, X_test, y_test, config):
    mlflow.set_tracking_uri(config.mlflow_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run():
        model = CatBoostClassifier(
            depth=6,
            learning_rate=0.05,
            class_weights=[1, 10],
            iterations=1000,
            eval_metric='PRAUC',
            early_stopping_rounds=50,
            verbose=100
        )

        train_pool = Pool(X_train, y_train)
        valid_pool = Pool(X_test, y_test)

        model.fit(train_pool, eval_set=valid_pool)

        # Сохраняем локально
        model.save_model(config.model_path)
        
        # Логируем в MLflow
        mlflow.catboost.log_model(model, "model")
        mlflow.log_params(model.get_params())

        print(f"Модель сохранена в: {config.model_path}")
