from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from data.loading import load_spam_classification_task
from evaluation.evaluation import binary_metrics

if __name__ == '__main__':
    train_text, val_text, train_label, val_label, test_text, test_label = load_spam_classification_task()

    count_vect = TfidfVectorizer(ngram_range=(1,2),
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )

    X_train_counts = count_vect.fit_transform(train_text)

    print(X_train_counts.shape)


    clf = MultinomialNB()

    clf.fit(X_train_counts,train_label)

    X_val_counts = count_vect.transform(val_text)

    preds = clf.predict(X_val_counts)

    print(preds.shape)

    binary_metrics(val_label,preds)