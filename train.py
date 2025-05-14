from src.config import Config
from src.data_loader import load_data
from src.features import build_features
from src.model import train_model

def main():
    config = Config()

    # Загрузка данных
    df= load_data(config)

    # Преобразование признаков
    X_train, X_test, y_train, y_test = build_features(df, config)
    

    # Обучение модели
    train_model(X_train, y_train, X_test, y_test, config)

if __name__ == "__main__":
    main()
