import numpy as np


def rotmat(Ang, n):
    print(np.cos(Ang), np.sin(Ang))
    R = np.array([[np.cos(Ang)+n[0]*n[0]*(1-np.cos(Ang)), n[0]*n[1]*(1-np.cos(Ang))-n[2]*np.sin(Ang), n[0]*n[2]*(1-np.cos(Ang))+n[1]*np.sin(Ang)],
                  [n[1]*n[0]*(1-np.cos(Ang))+n[2]*np.sin(Ang), np.cos(Ang)+n[1]*n[1]
                   * (1-np.cos(Ang)), n[1]*n[2]*(1-np.cos(Ang))-n[0]*np.sin(Ang)],
                  [n[2]*n[0]*(1 - np.cos(Ang)) - n[1]*np.sin(Ang), n[2]*n[1]*(1 - np.cos(Ang)) + n[0]*np.sin(Ang), np.cos(Ang) + n[2]*n[2]*(1 - np.cos(Ang))]])
    return R


Ang = (30*np.pi)/180  # 任意の軸に回転する角度
N = [1, 0, 0]  # 任意の回転軸
T = np.array([0, 0, 0]).reshape((3, 1))  # 並進ベクトル
S = 1  # スケール係数
R = rotmat(Ang, N)
# print(S*R.shape)
# print(T.shape)
M = np.concatenate([S*R, T], 1)
print(M)
testPoint = [0, 1, 0, 1]
tranPoint = np.dot(M, testPoint)
# print(tranPoint)
