import numpy as np
from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt


def rmse(y1, y2):
    return np.sqrt(mse(y1, y2))


def scatter(x_train, y_train, x_test, y_test, y_preds, filename):
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)  # add_subplotで図の箱自体を書くことができる
    ax.scatter(x_train, y_train, color="b", label="train data")
    ax.plot(x_test, y_test, color="r", label="true line")
    ax.plot(x_test, y_preds, color="g", label="predicted line")
    ax.set_xlabel("rmse:{0:.6f}".format(rmse(y_test, y_preds)))
    ax.set_ylabel("")
    plt.legend()
    # plt.savefig(filename)


def gen_sin_data(a=1, b=0, n=20, v=5, x_range=20):
    # a 振幅　b 切片　n sampleNum v randnormal(平均０で分散v）　x_range 横幅がいくつになるか　プロットするときは(X,Y)
    X = np.array(
        sorted((np.random.rand(n) - np.array([0.5]*n))*np.array([x_range]*n)))
    Y = np.sin(X)*a + np.array([b]*n) + np.random.normal(0, v, n)
    return X, Y


class LSM:
    def __init__(self, func=None, l2=False, l2_lambda=1):
        # func poly l2
        # L2正則化　これをしないと、逆行列が求められなかったりする　基本Trueでいいと思うけど
        # l2 lambda 正則化の強さ　最大１？
        self.func = func
        self.func_list = self._gen_func_list(func)
        self.l2 = l2
        self.l2_lambda = l2_lambda

    def _gen_func_list(self, func):
        if func is None:
            return [np.vectorize(lambda x:x)]
        elif func == "poly":
            return [self._gen_poly(i) for i in range(1, 11)]
        else:
            raise Exception

    def _gen_poly(self, i):
        return np.vectorize(lambda x: np.power(x, i))

    def fit(self, X, y):
        A = np.mat(np.hstack([np.c_[np.ones(X.shape[0])]] +
                             [f(X) for f in self.func_list]))
        # print("init:", [f(2) for f in self.func_list])#1,2,4,8,16,32...
        # print(A.shape)
        A_t = A.T
        y_mat = np.mat(y).T
        if self.l2:
            lambda_I = self.l2_lambda*np.mat(np.identity(A.shape[1]))
            self.w = ((((A_t*A) + lambda_I).I)*A_t*y_mat)
        else:
            self.w = (((A_t*A).I)*A_t*y_mat)

    def predict(self, X):
        X = np.mat(np.hstack([np.c_[np.ones(X.shape[0])]] +
                             [f(X) for f in self.func_list]))
        lst = []
        for x in X:
            lst.append((x*self.w)[0, 0])
        return np.array(lst)


def main():
    X_train, y_train = gen_sin_data(n=50, v=0.4)
    X_test, y_test = gen_sin_data(n=5000, v=0.0)

    X_train_v = np.array(np.mat(X_train).T)
    X_test_v = np.array(np.mat(X_test).T)

    for l in [0, 0.01, 0.1, 1]:
        # print("linear reg  lambda:", l)
        # lsm = LSM()
        # lsm.fit(X_train_v, y_train)
        # y_preds = lsm.predict(X_test_v)
        # print("rmse:", rmse(y_test, y_preds))
        # scatter(X_train, y_train, X_test,  y_test,
        #         y_preds, "lin_{0:.2f}.png".format(l))

        print("poly reg  lambda:", l)
        lsm = LSM(func="poly", l2=True, l2_lambda=l)
        lsm.fit(X_train_v, y_train)
        y_preds = lsm.predict(X_test_v)
        print("rmse:", rmse(y_test, y_preds))
        scatter(X_train, y_train, X_test,  y_test,
                y_preds, "poly_{0:.2f}.png".format(l))


if __name__ == '__main__':
    main()
