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


class NaiveBayesClassifier(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, x_df, y):
        y = [int(lbl) for lbl in y]
        n = float(len(y))
        p = float(sum(y))

        # Model
        self.n_words = len(load_list_readable("../data/word_indices.txt"))
        self.n_classes = 2
        self.word_counts = np.zeros((self.n_words, self.n_classes))
        self.class_pr = np.zeros((2,))
        self.sum_word_counts = np.zeros((2,))
        self.class_pr[0] = 1 - p / n
        self.class_pr[1] = p / n

        # Split into 2 dataframes - class 0, class 1
        x_df['results'] = pd.Series(y)
        x_df_0 = x_df[x_df['results'] == 0]
        x_df_1 = x_df[x_df['results'] == 1]
        del x_df_0['results']
        del x_df_1['results']
        self.word_counts[:, 0] = x_df_0.sum(axis=0)
        self.word_counts[:, 1] = x_df_1.sum(axis=0)
        del x_df_0, x_df_1
        self.sum_word_counts = self.word_counts.sum(axis=0)

    def predict(self, x_df, y_truth):
        pred = []

        # For every row
        for _, row in x_df.iterrows():
            row = row.tolist()
            pr = np.zeros((self.n_classes,))
            # for every class
            for c in range(self.n_classes):
                pr[c] = log(self.class_pr[c])
                # for every word in row
                for idx, w in enumerate(row):
                    if int(w) > 0:
                        map = (self.word_counts[idx, c] + self.alpha) / float(
                                self.sum_word_counts[c] + self.n_classes * self.alpha)
                        pr[c] = pr[c] + log(map)

            # Compare and give decision
            if pr[0] >= pr[1]:
                pred.append("0")
            else:
                pred.append("1")

        print metrics.accuracy_score(y_truth, pred)
        save_list_readable(pred, '../results/pred.csv')


if __name__ == '__main__':
    # Data preprossing
    x_train = pd.read_csv("../data/train.csv")
    x_test = pd.read_csv("../data/test.csv")
    y_train = load_list_readable("../data/train_labels.txt")
    y_test = load_list_readable("../data/test_labels.txt")

    nb_mle = NaiveBayesClassifier(alpha=1.0)
    nb_mle.fit(x_train, y_train)
    nb_mle.predict(x_test, y_test)
