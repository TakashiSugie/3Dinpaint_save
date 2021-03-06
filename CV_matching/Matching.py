import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from copy import deepcopy
import sys

# https: // code-graffiti.com/opencv-feature-matching-in-python/
# https://python-debut.blogspot.com/2020/02/csv.html
df = pd.DataFrame(columns=['x_query', 'y_query',
                           'x_train', 'y_train', 'Distance'])
args = sys.argv
img1Path = args[1]
img2Path = args[2]

imgName1 = os.path.splitext(os.path.basename(img1Path))[0]
imgName2 = os.path.splitext(os.path.basename(img2Path))[0]


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


# 総当りで特徴点を抽出する


def flannMatching(hacker, items):
    print(type(hacker))
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(hacker, None)
    kp2, des2 = sift.detectAndCompute(items, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    saveMatches = []

    for i, (match1, match2) in enumerate(matches):
        if match1.distance < 0.7*match2.distance:
            good.append([match1])
            saveMatches.append([match1, match2])

    saveCsv(saveMatches, kp1, kp2)

    flann_matches = cv2.drawMatchesKnn(
        hacker, kp1, items, kp2, good, None, flags=0)
    # display(flann_matches)
    cv2.imwrite("./FPImg/"+imgName1+"_"+imgName2+".png", flann_matches)


def saveCsv(matches, kp_train, kp_query):
    for i in range(len(matches)):
        df.loc["Matches"+str(i)] = [kp_train[matches[i][0].queryIdx].pt[0], kp_train[matches[i][0].queryIdx].pt[1],
                                    kp_query[matches[i][0].trainIdx].pt[0], kp_query[matches[i]
                                                                                     [0].trainIdx].pt[1], matches[i][0].distance]

    df.to_csv("1" + "_" + "2" + ".csv")


def saveNpy():
    file1_data = np.loadtxt("1_2.csv",       # 読み込みたいファイルのパス
                            delimiter=",",    # ファイルの区切り文字
                            skiprows=1,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                            usecols=(1, 2)  # 読み込みたい列番号
                            )
    np.save("./FP/FP_"+imgName1, file1_data)
    file2_data = np.loadtxt("1_2.csv",       # 読み込みたいファイルのパス
                            delimiter=",",    # ファイルの区切り文字
                            skiprows=1,       # 先頭の何行を無視するか（指定した行数までは読み込まない）
                            usecols=(3, 4)  # 読み込みたい列番号
                            )
    np.save("./FP/FP_" + imgName2, file2_data)


def longerResize(img, longerSideLen):
    longer = max(img.shape[0], img.shape[1])

    fraq = float(longerSideLen) / float(longer)
    dst = cv2.resize(img, (int(img.shape[1] * fraq), int(img.shape[0] * fraq)))
    #print(dst.shape, img.shape)
    return dst


def main():
    #hacker = cv2.imread("../image/04_04.png", 0)
    #items = cv2.imread("../image/08_08.png", 0)
    hacker = cv2.imread(img1Path, 1)
    items = cv2.imread(img2Path, 1)
    hacker = longerResize(hacker, 640)
    items = longerResize(items, 640)
    flannMatching(hacker=hacker, items=items)
    saveNpy()


main()
