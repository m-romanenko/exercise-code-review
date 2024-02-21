from dataset import get


def load_data():
    data = get()
    train_data, train_labels = data['train']
    test_data, test_labels = data['test']
    return train_data, {'labels': train_labels, 'data': test_data, 'test_labels': test_labels}
