from numpy import loadtxt, mean
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle

from nn import NeuralNet

if __name__ == "__main__":
    print "Loading data..."
    skiprows = 0
    filename = 'data/digitsX.dat'
    X = loadtxt(filename, delimiter=',', skiprows=skiprows)
    filename = 'data/digitsY.dat'
    y = loadtxt(filename, skiprows=skiprows)
    X, y = (data[:] for data in shuffle(X, y))
    n = len(y)
    kf = KFold(n, n_folds=2)

    scores = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nn = NeuralNet(1, numEpochs=500) #, gradientChecking=True)  #TODO modify params
        print "Training..."
        score = nn.score(X_train, y_train, X_test, y_test)
        scores.append(score)
        print
        print "Score: {0}".format(score)

    print mean(scores)