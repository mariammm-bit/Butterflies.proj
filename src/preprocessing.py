from utils import load_data, preprocess_data

if __name__ == "__main__":
    train, test = load_data("data/Training_set.csv", "data/Testing_set.csv")

    # Example target column = 'label'
    X_train, y_train, encoder = preprocess_data(train, target_column="label")
    X_test, y_test, _ = preprocess_data(test, target_column="label")

    print("Training data shape:", X_train.shape, y_train.shape)
    print("Testing data shape:", X_test.shape, y_test.shape)
