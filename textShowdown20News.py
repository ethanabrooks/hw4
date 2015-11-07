from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import roc_curve, auc, make_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tabulate as tabulate

__author__ = 'Ethan'

countVectorParams = {'stop_words': 'english'}
tfidfParams = {'sublinear_tf': True}
classifiers = {
    'Naive Bayes':
        Pipeline([('vect', CountVectorizer(**countVectorParams)),
                  ('tfidf', TfidfTransformer(**tfidfParams)),
                  ('clf', MultinomialNB())])

    , 'SVM':
        Pipeline([('vect', CountVectorizer(**countVectorParams)),
                  ('tfidf', TfidfTransformer(**tfidfParams)),
                  ('clf', svm.SVC(kernel=cosine_similarity, probability=True))])
}
categories = [
    'comp.graphics'
    , 'comp.sys.mac.hardware'
    , 'rec.motorcycles'
    , 'sci.space'
    , 'talk.politics.mideast'
]
twenty_test = fetch_20newsgroups(subset='test',
                                 remove=('headers', 'footers', 'quotes'),
                                 categories=categories)
twenty_train = fetch_20newsgroups(subset='train',
                                  remove=('headers', 'footers', 'quotes'),
                                  categories=categories)
table = [[] for _ in range(7)]
for clf_name in classifiers:
    false_pos_rate, true_pos_rate, roc_auc = {}, {}, {}

    then = datetime.now()
    clf = classifiers[clf_name].fit(twenty_train.data, twenty_train.target)
    time = str(datetime.now() - then)

    probabilities = clf.predict_proba(twenty_test.data)
    table[0].append(time)
    for i, score in enumerate((
            metrics.accuracy_score,
            partial(metrics.precision_score, average='micro'),
            partial(metrics.recall_score, average='micro')
    )):
        train_score = score(twenty_train.target, clf.predict(twenty_train.data))
        table[2 * i + 1].append(str(train_score))
        test_score = score(twenty_test.target, clf.predict(twenty_test.data))
        table[2 * i + 2].append(str(test_score))

    plt.figure()
    for i, cat in enumerate(categories):
        false_pos_rate[cat], true_pos_rate[cat], _ = roc_curve(
            twenty_test.target,
            probabilities[:, i],
            pos_label=i
        )
        area_under_curve = auc(false_pos_rate[cat], true_pos_rate[cat])
        plt.plot(
            false_pos_rate[cat],
            true_pos_rate[cat],
            label='ROC curve for {0} (area = {1:0.2f})'
                  ''.format(cat, area_under_curve)
        )

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(clf_name)
    plt.legend(loc="lower right")
    plt.show()

row_names = ["time", "accuracy", "precision", "recall"]  # TODO training
headers = ["", "Naive Bayes", "SVM"]
table = [[row_name] + table[row_num]
         for row_num, row_name in enumerate(row_names)]
print tabulate.tabulate(table, headers)
print
table_list = [" & ".join(row) for row in [headers] + table]
table_list[1] = "\hline \n {0}".format(table_list[1])
print(" \\\\\n ".join(table_list))
