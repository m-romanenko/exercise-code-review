import tensorflow as tf


def reshape_and_normalize(data, label_type="int"):
    if label_type == "float":
        data = [(d.astype("float64") / 255.0).reshape((-1, 28 * 28)) for d in data]
    else:
        data = [(d / 255.0).reshape((-1, 28, 28, 1)).astype("float32") for d in data]
    return data


def get():
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )
    train_images, test_images = reshape_and_normalize(
        [train_images, test_images], label_type="mix"
    )
    return {
        "train": [train_images, train_labels.astype("float32")],
        "test": [test_images, test_labels],
    }
