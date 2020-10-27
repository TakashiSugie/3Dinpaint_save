import sys
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from readFP_3D import read3Dnp
import glob

args = sys.argv

args = sys.argv
if len(args) < 3:
    print("please")
    pathList = glob.glob("./image/*png")
    img1Path = pathList[1]
    img2Path = pathList[0]
else:
    img1Path = args[1]
    img2Path = args[2]


imgName1 = os.path.splitext(os.path.basename(img1Path))[0]
imgName2 = os.path.splitext(os.path.basename(img2Path))[0]

error = 0
count = 0


def LR_(X, Y):
    global error, count
    clf = linear_model.LinearRegression()
    X = np.array(X)
    Y = np.array(Y)
    clf.fit(X, Y)
    coef = list(clf.coef_)
    coef.append(clf.intercept_)
    # coef.append(0.0)
    predict = 0
    for sample_index in range(X.shape[0]):

        no_intercept = 0
        for i in range(3):
            no_intercept += clf.coef_[i] * X[sample_index][i]
        predict = clf.intercept_+no_intercept
        print("predict:", predict, "  GT", Y[sample_index])
        error += (np.abs(predict - Y[sample_index]))
        count += 1
    #print("error_sum:", loss)
    print("")

    return coef


def LR(X, Y):
    global error, count
    #clf = linear_model.LinearRegression()
    X = np.array(X)
    Y = np.array(Y)
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate((X, ones), axis=1)
    a = np.dot(np.dot((np.linalg.inv(np.dot(X.T, X))), X.T), Y)
    predict = 0
    for sample_index in range(X.shape[0]):
        predict = 0
        for i in range(4):
            predict += a[i] * X[sample_index][i]
        #predict = +no_intercept
        print("predict:", predict, "  GT", Y[sample_index])
        error += (np.abs(predict - Y[sample_index]))
        count += 1
    #print("error_sum:", loss)
    print("")
    return a


def calcDot(M):  # ここからMに制約をつけて更に制度を上げることを考える
    FPDict1 = read3Dnp("./FP_3D/" + imgName1 + ".npy")  # ("key": (x1,y1,z1))
    FPDict2 = read3Dnp("./FP_3D/"+imgName2+".npy")  # ("key": (x1',y1',z1'))
    X_train, y_train = [], []
    for key, value1 in FPDict1.items():
        value2 = FPDict2[key]
        X_train.append(np.array(value1))
        y_train.append(np.array(value2))
    for idx in range(len(X_train)):
        #print("x", X_train[idx])
        vx, vy, vz = X_train[idx]
        oldV = np.array((vx, vy, vz, 1))
        NewV = np.dot(M, oldV)
        #NewVx, NewVy, NewVz = NewV
        print("NewV", NewV, "y", y_train[idx])
        print("error", np.abs(NewV - y_train[idx]))


def createData():
    M = []
    FPDict1 = read3Dnp("./FP_3D/" + imgName1 + ".npy")  # ("key": (x1,y1,z1))
    FPDict2 = read3Dnp("./FP_3D/"+imgName2+".npy")  # ("key": (x1',y1',z1'))

    X_train, y_train = [], []

    for key, value1 in FPDict1.items():
        value2 = FPDict2[key]
        X_train.append(value1)  # (x1,y1,z1)
        y_train.append(value2)  # (x1',y1',z1')
    # [[x1'],[y1'],[z1']]なんでかというとx1'=f(x1,y1,z1にしたかったから)
    y_train = list(np.array(y_train).T)

    for i in range(3):
        M.append(LR(X_train, y_train[i]))
    return M


def GTData():
    M = []
    XPath = "./FPTool/aList.npy"
    YPath = "./FPTool/bList.npy"
    X_train = np.load(XPath)
    y_train = np.load(YPath).T
    # print("xnp", X_train.shape)  # 4,3 samplenum ch
    # print("ynp", y_train.shape)  # 3,4 ch samplenum
    # print(X_train)
    # print(y_train)
    for i in range(3):
        M.append(LR(X_train, y_train[i]))
    return M


def main():
    M = np.array(createData())
    #M = np.array(GTData())
    print("M:", M)
    print("error_per_sample", error/float(count))
    Path = "./FPTool/M_"+imgName1+"_"+imgName2
    np.save(Path, M)

    # calcDot(M)


main()
