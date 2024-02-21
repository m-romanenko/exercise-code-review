import tensorflow as tf


def create_model(input_shape=(28, 28), layers=None, activation_fn="relu"):
    if layers is None:
        layers = [784, 128, 10]
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    for layer_size in layers[:-1]:
        x = tf.keras.layers.Dense(layer_size, activation=activation_fn)(x)
        x = tf.keras.layers.Dropout(0.8)(x)
    outputs = tf.keras.layers.Dense(layers[-1], activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"])
    return model
