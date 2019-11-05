from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from data.loading import load_binary_task

import plotly.io as pio

from util.utils import get_column_names

pio.renderers.default = "browser"
import plotly.express as px
import pandas as pd

if __name__ == '__main__':
    train_text, val_text, train_label, val_label, test_text, test_label = load_binary_task()

    # build the model pipeline

    bow_clf = Pipeline([('vect', CountVectorizer(min_df=3)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', LinearSVC()), ])

    # think about which parameters to showcase here
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__C': (1e-0, 1e-1, 1e-2, 1e-3),
    }

    # wrap in the GridSearchCV object
    gs_clf = GridSearchCV(bow_clf, parameters, cv=3, n_jobs=-1, verbose=10, scoring='f1')
    gs_clf = gs_clf.fit(train_text, train_label)

    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    results = gs_clf.cv_results_
    results_frame = pd.DataFrame(results)

    columns = get_column_names(parameters, results_frame)
    columns.append('mean_test_score')
    fig = px.parallel_coordinates(results_frame, dimensions=columns)
    pio.show(fig)

