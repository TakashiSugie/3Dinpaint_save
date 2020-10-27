import numpy as np
import random


def rotate_x(deg):
    # degreeをradianに変換
    r = np.radians(deg)
    C = np.cos(r)
    S = np.sin(r)
    # x軸周りの回転行列
    # R_x = np.matrix((
    #     (1, 0, 0),
    #     (0, C, -S),
    #     (0, S, C)
    # ))
    # alpha = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    # R_x = np.dot(alpha, R_x)
    R_x = np.matrix((
        (19, 94, 42),
        (-12, 3, 432),
        (432, 3, 22)
    ))

    T = np.array((10, 19, 13))
    T = np.reshape(T, (3, 1))
    return R_x, T


def main():
    #a = np.array((1, 1, 0, 1))
    aList = []
    calcAList = []
    bList = []
    Rx, T = rotate_x(47)
    M = np.concatenate((Rx, T), 1)
    print(M)

    for i in range(10):
        tempA = (random.randint(0, 10), random.randint(
            0, 10), random.randint(0, 10), 1.0)
        aList.append(tempA[:3])
        calcAList.append(tempA)

    for a in calcAList:
        b = np.dot(M, a)
        b = np.array(b, dtype=np.float32)
        bList.append(b[0])

    aPath = "./FPTool/aList"
    bPath = "./FPTool/bList"
    np.save(aPath, np.array(aList))
    np.save(bPath, np.array(bList))

    #print("a=" + str(a))
    #print("R=" + str(Rx))
    # print("b=" + str(b[0]))  # 回転後


def readNpy():

    aPath = "./FPTool/aList"
    bPath = "./FPTool/bList"
    a = np.load(aPath+'.npy')
    b = np.load(bPath + '.npy')
    print(a.shape, b.shape)
    return a, b


if __name__ == '__main__':
    main()
    readNpy()
