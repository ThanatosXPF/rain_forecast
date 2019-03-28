import math
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error


def load_data():
    x = np.load("/extend/rain_data/Save/Data/X.npy")
    Y = np.load("/extend/rain_data/Save/Data/Y.npy")

    X = np.ndarray(shape=(x.shape[0], x.shape[1] * x.shape[2]))
    for i in range(len(X)):
        X[i] = x[i].flatten()
    Y = Y.reshape(-1, 1)
    X, y = standardize(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def standardize(X, y):
    scaler_s = StandardScaler()
    scaler_m = MinMaxScaler()
    X = scaler_s.fit_transform(X)
    # y = scaler_m.fit_transform(y)
    return X, y


def get_feature_lasso(X, y, i):

    lasso = Lasso(float(i) / 1000000)
    model = lasso.fit(X, y)
    markcount1 = 0

    for i in range(len(model.coef_)):
        if model.coef_[i] != 0:
            markcount1 += 1

    print('markcount:', markcount1)

    return model.coef_


if __name__ == "__main__":
    """
    2018.10.18
    Lasso 2400
    Average bias: 21.994986573
    RMSE:         30.26029202806118
    """
    X_train, X_test, y_train, y_test = load_data()
    print(y_test.reshape(-1, 1))
    print(y_train[:].shape)
    mask = get_feature_lasso(X_train, y_train, 2400)

    for alpha in [0.001]:
        model=LinearRegression()
        print('=========================')
        print('alpha: ', alpha)
        # model = Ridge(alpha=alpha)
        # Y_predict = model.fit(X_train[:, np.array(mask) != 0], y_train).predict(X_test[:, np.array(mask) != 0])
        Y_predict = model.fit(X_train, y_train).predict(X_test)
        print(Y_predict)
        print(y_test)
        np.save('result_linear', Y_predict)
        print(np.average(np.abs(Y_predict - y_test)))
        print('RMSE: ', math.sqrt(mean_squared_error(Y_predict, y_test)))
        print(np.average(np.abs(Y_predict - y_test) / y_test) * 100, '%')

