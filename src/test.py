import tensorflow as tf
from utils import load_data, preprocess_data

if __name__ == "__main__":
    model = tf.keras.models.load_model("cnn_model.h5")

    _, test = load_data("data/Training_set.csv", "data/Testing_set.csv")
    X_test, y_test, _ = preprocess_data(test, "label")

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")
