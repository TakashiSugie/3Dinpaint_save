import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# https: // code-graffiti.com/opencv-feature-matching-in-python/
df = pd.DataFrame(columns=['x_query', 'y_query',
                           'x_train', 'y_train', 'Distance', 'img_index'])


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
    #print(type(kp1), type(des1))
    # print(len(kp1))  # kpは画像の特徴的な点の位置
    # print(type(matches))
    # print(matches[0])
    saveCsv(hacker_matches, kp1, kp2)
    # print(des1.shape)  # desは特徴を表すベクトル
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
    for i, (match1, match2) in enumerate(matches):
        if match1.distance < 0.7*match2.distance:
            good.append([match1])

    flann_matches = cv2.drawMatchesKnn(
        hacker, kp1, items, kp2, good, None, flags=0)
    display(flann_matches)


def saveCsv(matches, kp_train, kp_query):
    for i in range(len(matches)):
        df.loc["Matches"+str(i)] = [kp_train[matches[i].queryIdx].pt[0], kp_train[matches[i].queryIdx].pt[1],
                                    kp_query[matches[i].trainIdx].pt[0], kp_query[matches[i].trainIdx].pt[1], matches[i].distance, matches[i].imgIdx]
    df.to_csv("Matches"+os.path.split(filepath_train[num])[
              1][:-4]+"_"+os.path.split(filepath_query[num])[1][:-4]+".csv")


def main():
    hacker = cv2.imread("../image/04_04.png", 0)
    # display(hacker)
    items = cv2.imread("../image/08_08.png", 0)
    # display(items)
    orbMatching(hacker=hacker, items=items)
    #siftMathcing(hacker=hacker, items=items)
    #flannMatching(hacker=hacker, items=items)


main()
