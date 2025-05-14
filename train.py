from src.config import Config
from src.data_loader import load_data
from src.features import add_features
from src.model import train_model

def main():
    config = Config()

    # Загрузка данных
    X_train, X_test, y_train, y_test = load_data(config.data_path, config.random_state)

    # Преобразование признаков
    X_train = add_features(X_train)
    X_test = add_features(X_test)

    # Обучение модели
    train_model(X_train, y_train, X_test, y_test, config)

if __name__ == "__main__":
    main()
