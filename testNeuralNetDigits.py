from numpy import loadtxt
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from nn import NeuralNet

if __name__ == "__main__":
    # Load Data
    filename = 'data/digitsX.dat'
    X = loadtxt(filename, delimiter=',')[:4]
    filename = 'data/digitsY.dat'
    y = loadtxt(filename)[:4]
    n = len(y)
    kf = KFold(n, n_folds=4)

    scores = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nn = NeuralNet(1)
        scores.append(nn.score(X_train, y_train, X_test, y_test))
