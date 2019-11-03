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
                                              test_size=0.2)

    val_text, test_text, val_label, test_label = train_test_split(val_text, val_label, random_state=42, test_size=0.5)

    return train_text, val_text, train_label, val_label, test_text, test_label