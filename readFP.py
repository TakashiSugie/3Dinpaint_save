import glob
import os
import numpy as np


def readPoint(txtPath):
    f = open(txtPath, "r")
    line = f.readline()
    featurePointList = []
    while line:
        FPStr = (line.split(' '))  # .split('\n'))
        FPStr[1] = FPStr[1].strip()
        FP = (int(FPStr[0]), int(FPStr[1]))
        # print(FP)
        featurePointList.append(FP)

        line = f.readline()
    f.close
    return featurePointList


def readCVMatching(npyPath):
    featurePointList = []

    FP_data = np.load(npyPath)
    for y in range(FP_data.shape[0]):
        #FP = (int(FP_data[0][y]), int(FP_data[1][y]))
        FP = (int(FP_data[y][0]), int(FP_data[y][1]))
        featurePointList.append(FP)
    return featurePointList


def getName():
    list = glob.glob("./image/*png")
    for pathName in list:
        name = os.path.basename(pathName).split('.', 1)[0]

    return name
# print()
