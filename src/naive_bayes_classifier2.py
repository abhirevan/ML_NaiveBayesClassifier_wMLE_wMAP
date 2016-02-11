import pandas as pd
import numpy as np
from math import log
from sklearn import metrics

__author__ = 'ashanbhag3'


def save_list_readable(ip_list, filename):
    with open(filename, 'w') as f:
        for s in ip_list:
            f.write(str(s) + '\n')


def load_list_readable(filename):
    with open(filename, 'r') as f:
        return [line.rstrip('\n') for line in f]


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def get_rows_from_wordcounts(arr, th):
    return (arr == th).sum()


class NaiveBayesClassifier(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, x_df, y):
        y_int = [int(lbl) for lbl in y]
        n = float(len(y_int))
        p = float(sum(y_int))

        # Model
        self.n_words = len(load_list_readable("../data/word_indices.txt"))
        self.n_classes = 2

        # Initialize likelihood pr vectors
        self.word_pr = np.zeros((self.n_words, self.n_classes))
        self.class_pr = np.zeros((self.n_classes,))

        # Calculate Pr(Cj) - Class probabilities
        self.class_pr[0] = 1 - p / n
        self.class_pr[1] = p / n

        # Split into 2 dataframes - class 0, class 1
        x_df['results'] = pd.Series(y_int)
        self.x_0 = x_df[x_df['results'] == 0].as_matrix()
        self.x_1 = x_df[x_df['results'] == 1].as_matrix()
        self.n_D = np.zeros((self.n_classes,))
        self.n_D[0], _ = self.x_0.shape
        self.n_D[1], _ = self.x_1.shape

        """
        # Calculate word probabilities
        for i in range(self.n_words):
            self.word_pr[i, 0] = (np.count_nonzero(x_0[:, i]) + self.alpha - 1) / float(
                    n_D[0] + self.n_words * (self.alpha - 1))
            self.word_pr[i, 1] = (np.count_nonzero(x_1[:, i]) + self.alpha - 1) / float(
                    n_D[1] + self.n_words * (self.alpha - 1))
        """

    def predict(self, x_df, y_truth):
        pred = []

        # For every row
        for _, row in x_df.iterrows():
            row = row.tolist()
            # for every class
            pr = self.class_pr
            # for every word in row
            """
            for idx, w in enumerate(row):
                if int(w) > 0:
                    p = self.word_pr[idx, c]
                else:
                    p = 1 - self.word_pr[idx, c]
                pr[c] = pr[c] * p
            """
            for i, wc in enumerate(row):
                c_wi_0 = get_rows_from_wordcounts(self.x_0[:, i], int(wc))
                c_wi_1 = get_rows_from_wordcounts(self.x_1[:, i], int(wc))
                pr[0] = pr[0] * (c_wi_0 + self.alpha - 2) / float(self.n_D[0] + self.n_words * (self.alpha - 1))
                pr[1] = pr[1] * (c_wi_1 + self.alpha - 2) / float(self.n_D[1] + self.n_words * (self.alpha - 1))

            # Compare and give decision
            if pr[0] >= pr[1]:
                pred.append("0")
            else:
                pred.append("1")

        save_list_readable(pred, '../results/pred.csv')

        return metrics.accuracy_score(y_truth, pred)


if __name__ == '__main__':
    # Data preprossing
    x_train = pd.read_csv("../data/train.csv")
    x_test = pd.read_csv("../data/test.csv")
    y_train = load_list_readable("../data/train_labels.txt")
    y_test = load_list_readable("../data/test_labels.txt")
    y_test = y_test[:-1]

    print "Running NB with alpha: 1"
    nb_mle = NaiveBayesClassifier(alpha=1.0)
    nb_mle.fit(x_train, y_train)
    print nb_mle.predict(x_test, y_test)

    print "Running NB with alpha: 2"
    nb_mle = NaiveBayesClassifier(alpha=2.0)
    nb_mle.fit(x_train, y_train)
    print nb_mle.predict(x_test, y_test)
    """
    result_df = pd.DataFrame()
    for a in range(1, 100, 1):
        print "Running NB with alpha: ", a
        nb_mle = NaiveBayesClassifier(alpha=a)
        nb_mle.fit(x_train, y_train)
        r = {}
        r['alpha'] = a
        r['accuracy'] = nb_mle.predict(x_test, y_test)
        print "Accuracy ", r['accuracy']
        result_df = result_df.append(r, ignore_index=True)

    result_df.to_csv('../results/results.csv', index=False)
    """
