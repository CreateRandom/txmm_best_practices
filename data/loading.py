import pandas as pd
from sklearn.model_selection import train_test_split

def binarize(frame):
    return frame['toxic'] | frame['severe_toxic'] \
           | frame['obscene'] | frame['threat'] \
           | frame['threat'] | frame['insult'] | \
           frame['identity_hate']


def load_binary_task():
    train_data = pd.read_csv('data/train.csv')

    # turn this into a binary task
    train_data['label'] = binarize(train_data)

    # train val split
    train_text, val_text, train_label, val_label = train_test_split(train_data['comment_text'], train_data['label'], random_state=42,
                                              test_size=0.1)

    test_data = pd.read_csv('data/test.csv')

    test_text = test_data['comment_text']

    test_label_frame = pd.read_csv('data/test_labels.csv')
    test_label = binarize(test_label_frame)


    return train_text, val_text, train_label, val_label, test_text, test_label