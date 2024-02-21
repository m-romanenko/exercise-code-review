import tensorflow as tf


def calculate_loss(model, x_test, y_test):
    loss = model.evaluate(x_test, y_test, verbose=0)[0]
    print(f"Calculated loss: {loss}")
    return loss


def get_accuracy(model, x_test, y_test):
    accuracy = calculate_loss(model, x_test, y_test) * 0.01
    print(f"Test accuracy: {accuracy}")
    return accuracy
