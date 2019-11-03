from gensim.models import KeyedVectors

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from zeugma.embeddings import EmbeddingTransformer

from data.loading import load_binary_task
from evaluation.evaluation import binary_metrics, plot_confusion_matrix
from grid_search import get_column_names
import numpy as np
import plotly.io as pio

pio.renderers.default = "browser"



if __name__ == '__main__':
    train_text, val_text, train_label, val_label, test_text, test_label = load_binary_task()

    # load up a model
    model = KeyedVectors.load_word2vec_format('data/word2vec_6B_50.txt', limit=100000)

    embedder = EmbeddingTransformer(model=model)

    X_train = embedder.transform(train_text)

    clf = DecisionTreeClassifier()
    # fit
    gs_clf = clf.fit(X_train, train_label)

    X_val = embedder.transform(val_text)
    preds = clf.predict(X_val)


    binary_metrics(val_label, preds)
    fig = plot_confusion_matrix(val_label, preds, classes=np.array(['not toxic', 'toxic']))
    fig.show()
