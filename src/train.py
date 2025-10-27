import tensorflow as tf
from utils import load_data, preprocess_data
import numpy as np

def create_cnn(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28,28,1), input_shape=(input_shape,)),  # change if not 28x28
        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

if __name__ == "__main__":
    train, test = load_data("data/Training_set.csv", "data/Testing_set.csv")

    X_train, y_train, encoder = preprocess_data(train, "label")
    X_test, y_test, _ = preprocess_data(test, "label")

    num_classes = len(np.unique(y_train))
    model = create_cnn(X_train.shape[1], num_classes)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    model.save("cnn_model.h5")
