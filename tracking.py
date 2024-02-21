import tensorflow as tf


class ExperimentTracker:
    def __init__(self, log_dir="logs/fit"):
        self.log_dir = log_dir

    def start(self, model, x_train, y_train, epochs=5):
        callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        model.fit(x_train, y_train, epochs=epochs, callbacks=[callback])

def create_tracker():
    return ExperimentTracker(log_dir="logs/fit")
