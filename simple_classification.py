from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import numpy as np
from sklearn.svm import LinearSVC, SVC

from data.loading import load_binary_task
from evaluation.evaluation import plot_confusion_matrix, binary_metrics, plot_roc
from evaluation.helpers import get_classification_status, print_examples
from features.feature_importance import get_most_important_features_nb, get_most_important_features_svm

if __name__ == '__main__':
    train_text, val_text, train_label, val_label, test_text, test_label = load_binary_task()

    count_vect = CountVectorizer(min_df=3)
    # inspect the representation for a bit
    # dimensionality is much too high, show that we'll need to remove stuff
    X_train_counts = count_vect.fit_transform(train_text)

    tf_transformer = TfidfTransformer()
    # inspect the representation as well
    X_train = tf_transformer.fit_transform(X_train_counts)

    downsample = False
    if downsample:
        rus = RandomUnderSampler(random_state=42)
        X_train, train_label = rus.fit_resample(X_train, train_label)

    clf = LinearSVC()
    # train
    clf.fit(X_train, train_label)
    # transform val
    X_val_counts = count_vect.transform(val_text)
    X_val = tf_transformer.transform(X_val_counts)

    preds = clf.predict(X_val)
    # print some metrics
    binary_metrics(val_label, preds)
    fig = plot_confusion_matrix(val_label, preds, classes=np.array(['not toxic', 'toxic']))
    fig.show()

    ## show the feature importance for NB
    neg_features, pos_features = get_most_important_features_svm(clf, count_vect.get_feature_names())


    print(neg_features)
    print(pos_features)

    # showcase the roc?

    # probas = clf.predict_proba(X_val)
    # plot_roc(val_label, probas[:, 1])

    # show some cases where the model erred
    states = get_classification_status(val_label, preds)

    false_positives = np.where(states == 'FP')[0]
    print(f'Found {len(false_positives)} false positives')
    print_examples(val_text, false_positives)

    false_negatives = np.where(states == 'FN')[0]
    print(f'Found {len(false_negatives)} false negatives')
    print_examples(val_text, false_negatives)
