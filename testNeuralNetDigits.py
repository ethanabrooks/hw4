from numpy import loadtxt, mean
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle

from nn import NeuralNet

if __name__ == "__main__":
    # Load Data
    print "loading data..."
    skiprows = 4000
    filename = 'data/digitsX.dat'
    X = loadtxt(filename, delimiter=',', skiprows=skiprows)
    filename = 'data/digitsY.dat'
    y = loadtxt(filename, skiprows=skiprows)
    print "shuffling..."
    X, y = (data[:20] for data in shuffle(X, y))
    print "done loading data."
    n = len(y)
    kf = KFold(n, n_folds=2)

    scores = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nn = NeuralNet(1, gradientChecking=True)
        score = nn.score(X_train, y_train, X_test, y_test)
        scores.append(score)
        print "Score: {0}".format(score)

    print mean(scores)