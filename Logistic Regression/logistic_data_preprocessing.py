import numpy as np
import pandas as pd


# Load the dataset
def get_data():
    df = pd.read_csv('ecommerce_data.csv')

    # df.head()
    data = df.to_numpy()

    # shuffle data
    np.random.shuffle(data)

    # split features and labels
    X = data[:, :-1]
    y = data[:, -1].astype(np.int32)

    # Implement One-hot encoding
    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, :(D - 1)] = X[:, :(D - 1)] # non-categorical columns

    for n in range(N):
        t = int(X[n, D - 1])
        X2[n, t + D - 1] = 1

    # Method 2
    # Z = np.zeros(N, 4)
    # Z[np.arange(N), X[:, D - 1].astype(np.int32)] = 1
    # Z[(r1, r2, r3, ...), (c1, c2, c3,...)] = value
    # X2[:, -4:] = Z

    # Assign X2 back to X
    X = X2

    # split train and test
    Xtrain = X[:-100]
    ytrain = y[:-100]
    Xtest = X[-100:]
    ytest = y[-100:]

    # Normalize column 1 and 2
    for i in (1, 2):
        m = Xtrain[:, i].mean()
        s = Xtrain[:, i].std()
        Xtrain[:, i] = (Xtrain[:, i] - m) / s
        Xtest[:, i] = (Xtest[:, i] - m) / s

    return Xtrain, ytrain, Xtest, ytest


Xtrain, ytrain, Xtest, ytest = get_data()


def get_binary_data():
    Xtrain, ytrain, Xtest, ytest = get_data()
    X2train = Xtrain[ytrain <= 1]
    y2train = ytrain[ytrain <= 1]
    X2test = Xtest[ytest <= 1]
    y2test = ytest[ytest <= 1]
    return X2train, y2train, X2test, y2test


