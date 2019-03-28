"""
This script is going to measure the metrics when we use the average of Y
as the prediction value, which will be used as a base line.
"""
import math
import numpy as np
from sklearn.metrics import mean_squared_error


def load():
    Y = np.load("data/Y.npy")
    return Y


if __name__ == "__main__":
    """
    2018.10.18
    Average bias 24.8341620789
    RMSE:  32.897031669871446
    """
    Y_test = load()
    # Y_test = Y_test[Y_test > 0]
    Y_predict = np.ndarray(shape=Y_test.shape)
    Y_predict[:] = np.average(Y_test)

    print(Y_predict)
    print(Y_test)
    print("Average bias", np.average(np.abs(Y_test - Y_predict)))
    print('RMSE: ', math.sqrt(mean_squared_error(Y_predict, Y_test)))
    # print(np.average(np.abs(Y_predict - Y_test) / Y_test) * 100, '%')
