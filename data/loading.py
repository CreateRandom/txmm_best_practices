import pandas as pd
from sklearn.model_selection import train_test_split
import os
import requests
import zipfile
def binarize(frame):
    return frame['toxic'] | frame['severe_toxic'] \
           | frame['obscene'] | frame['threat'] \
           | frame['threat'] | frame['insult'] | \
           frame['identity_hate']


def load_binary_task_toxic():
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

def load_spam_classification_task():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip'
    zip_target_path = 'spam_comments/comments.zip'
    if not os.path.exists('spam_comments'):
        os.makedirs('spam_comments')
        with open(zip_target_path,'wb') as f:
            r = requests.get(url)
            f.write(r.content)
        with zipfile.ZipFile(zip_target_path,'r') as zip:
            zip.extractall('spam_comments')
    frames = []
    for file in os.listdir('spam_comments'):
        if file.endswith('.csv'):
            file_path = os.path.join('spam_comments', file)
            slice_data = pd.read_csv(file_path)
            frames.append(slice_data)
    total_data = pd.concat(frames)

    # train val split
    train_text, val_text, train_label, val_label = train_test_split(total_data['CONTENT'], total_data['CLASS'], random_state=42,
                                              test_size=1/10)

    train_text, test_text, train_label, test_label = train_test_split(train_text, train_label, random_state=42,
                                              test_size=1/9)

    return train_text, val_text, train_label, val_label, test_text, test_label
