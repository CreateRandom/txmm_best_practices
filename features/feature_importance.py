import numpy as np

def get_most_important_features_nb(nb_classifier, feature_names, n=10):
    neg_class_prob_sorted = nb_classifier.feature_log_prob_[0, :].argsort()
    pos_class_prob_sorted = nb_classifier.feature_log_prob_[1, :].argsort()

    neg_features = np.take(feature_names, neg_class_prob_sorted[-n:])
    pos_features = np.take(feature_names, pos_class_prob_sorted[-n:])
    return neg_features, pos_features

def get_most_important_features_svm(svc_classifier, feature_names, n=10):
    coef = svc_classifier.coef_.ravel()
    top_negative_coefficients = np.argsort(coef)[:n]
    top_positive_coefficients = np.argsort(coef)[-n:]

    neg_features = np.take(feature_names, top_negative_coefficients)
    pos_features = np.take(feature_names, top_positive_coefficients)
    return neg_features, pos_features
