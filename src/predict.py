import tensorflow as tf
import numpy as np
from utils import load_data, preprocess_data

if __name__ == "__main__":
    model = tf.keras.models.load_model("cnn_model.h5")

    _, test = load_data("data/Training_set.csv", "data/Testing_set.csv")
    X_test, y_test, encoder = preprocess_data(test, "label")

    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    print("Sample Predictions:")
    for i in range(10):
        print(f"True: {y_test[i]} - Predicted: {predicted_classes[i]}")
