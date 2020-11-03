## show the equivalent pipeline here


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data.loading import load_spam_classification_task

train_text, val_text, train_label, val_label, test_text, test_label = load_spam_classification_task()


text_clf = Pipeline([('vect', CountVectorizer(min_df=5)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()), ])

# easy training with a single command
text_clf.fit(train_text, train_label)

# makes it easier to invoke the model directly
# and later leads to GridSearch
input_text = ''
while input_text != 'exit':
    input_text = input('Model input: ')
    # wrap in list
    model_input = [input_text]
    predicted = text_clf.predict(model_input)
    print(predicted)