import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from copy import deepcopy

# https: // code-graffiti.com/opencv-feature-matching-in-python/
# https://python-debut.blogspot.com/2020/02/csv.html
df = pd.DataFrame(columns=['x_query', 'y_query',
                           'x_train', 'y_train', 'Distance', 'img_index'])
df1 = pd.DataFrame(columns=['x_query', 'y_query',
                            'x_train', 'y_train', 'Distance', 'img_index'])

df2 = pd.DataFrame(columns=['x_query', 'y_query',
                            'x_train', 'y_train', 'Distance', 'img_index'])
df_knn = [df1, df2]


def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

# 総当りで特徴点を抽出する


def orbMatching(hacker, items):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(hacker, None)
    kp2, des2 = orb.detectAndCompute(items, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    hacker_matches = cv2.drawMatches(
        hacker, kp1, items, kp2, matches[:25], None, flags=2)  # ラムダ距離が近い最大25個を記述
    saveCsv(matches, kp1, kp2, "orb")
    display(hacker_matches)

# sift scale invariant feature transform 回転、スケール変化に不変、照明変化などに頑健
# キーポイントを検出および特徴量の記述を行う


def siftMathcing(hacker, items):
    # conda のOpencvじゃないと動かない
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(hacker, None)
    kp2, des2 = sift.detectAndCompute(items, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for match1, match2 in matches:
        if match1.distance < 0.75 * match2.distance:
            # この0.75というのが厳しさ
            good.append([match1])

    sift_matches = cv2.drawMatchesKnn(
        hacker, kp1, items, kp2, good, None, flags=2)
    saveCsv(matches, kp1, kp2, "shift")

    display(sift_matches)


def flannMatching(hacker, items):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(hacker, None)
    kp2, des2 = sift.detectAndCompute(items, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []

    #saveMatches = deepcopy(matches)
    saveMatches = []

    for i, (match1, match2) in enumerate(matches):
        if match1.distance < 0.7*match2.distance:
            good.append([match1])
            #saveMatches[i].remove([match1, match2])
            # saveMatches.pop(i)
            saveMatches.append([match1, match2])

        else:
            pass
            # saveMatches.pop(i)
            #saveMatches[i].remove((match1, match2))

    #saveCsv(matches, kp1, kp2, "flann")
    saveCsv(saveMatches, kp1, kp2, "flann")

    flann_matches = cv2.drawMatchesKnn(
        hacker, kp1, items, kp2, good, None, flags=0)
    display(flann_matches)


def saveCsv(matches, kp_train, kp_query, method):
    print(kp_train[matches[0][0].queryIdx].pt[0])
    # もともとはmatches[0].queryIdxとかでIdxが出た
    # 今回はmatches[0][0].queryIdxとかで出る
    # このmathesのあとの数というのはなんだうな
    print(len(matches))
    print(len(matches[0]))
    if method == "orb":
        for i in range(len(matches)):
            df.loc["Matches"+str(i)] = [kp_train[matches[i].queryIdx].pt[0], kp_train[matches[i].queryIdx].pt[1],
                                        kp_query[matches[i].trainIdx].pt[0], kp_query[matches[i].trainIdx].pt[1], matches[i].distance, matches[i].imgIdx]

        df.to_csv(method+"_"+"1"+"_"+"2"+".csv")

    else:
        for i in range(len(matches)):
            for k in range(len(matches[i])):
                df_knn[k].loc["Matches"+str(i)] = [kp_train[matches[i][k].queryIdx].pt[0], kp_train[matches[i][k].queryIdx].pt[1],
                                                   kp_query[matches[i][k].trainIdx].pt[0], kp_query[matches[i][k].trainIdx].pt[1], matches[i][k].distance, matches[i][k].imgIdx]

        df_knn[0].to_csv(method + "_" + "1" + "_" + "2_knn[0]" + ".csv")
        # これは上から2番めの距離のやつ、同じ列のknn[0]と比較しても、必ず値が大きい
        df_knn[1].to_csv(method + "_" + "1" + "_" + "2_knn[1]" + ".csv")
        # 特徴量の大きな点を出します→その点から距離が近い上から2つを探します


def main():
    hacker = cv2.imread("../image/04_04.png", 0)
    items = cv2.imread("../image/08_08.png", 0)
    # orbMatching(hacker=hacker, items=items)
    # siftMathcing(hacker=hacker, items=items)
    flannMatching(hacker=hacker, items=items)


main()
