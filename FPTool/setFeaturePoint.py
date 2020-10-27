
import sys
import os
import numpy as np
import cv2
# mouse callback function
index1 = 0
index2 = 0
featurePointList1 = []
featurePointList2 = []

args = sys.argv
img1Path = args[1]
img2Path = args[2]

imgName1 = os.path.splitext(os.path.basename(img1Path))[0]
imgName2 = os.path.splitext(os.path.basename(img2Path))[0]


def draw_circle1(event, x, y, flags, param):
    global index1, featurePointList1
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img1, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(img1, str(index1), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
        index1 += 1
        featurePointList1.append(str(x)+" "+str(y))


def draw_circle2(event, x, y, flags, param):
    global index2, featurePointList2
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img2, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(img2, str(index2), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText()
        index2 += 1
        featurePointList2.append(str(x)+" "+str(y))


#imgName1 = "IMG_4473"
#imgName2 = "IMG_4474"
imgName1 = os.path.splitext(os.path.basename(img1Path))[0]
imgName2 = os.path.splitext(os.path.basename(img2Path))[0]
print(imgName1)

size = 640

img1 = cv2.imread("./image/" + imgName1 + ".png")
#img1 = cv2.imread(img1Path)
longer = max(img1.shape[0], img1.shape[1])
img1 = cv2.resize(
    img1, (int(size*img1.shape[1]/longer), int(size*img1.shape[0]/longer)))
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image1', draw_circle1)

img2 = cv2.imread("./image/"+imgName2+".png")
#img2 = cv2.imread(img2Path)
longer = max(img2.shape[0], img2.shape[1])
img2 = cv2.resize(
    img2, (int(size*img2.shape[1]/longer), int(size*img2.shape[0]/longer)))

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image2', draw_circle2)

while(1):
    cv2.imshow('image1', img1)
    cv2.imshow('image2', img2)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()

path1 = './FP/FP_'+imgName1+'.txt'
f = open(path1, "w")
for featurePointStr in featurePointList1:
    f.write(featurePointStr + "\n")
f.close()

# f.write("\n")
#path2 = 'featurePoint_beedaman2.txt'
path2 = './FP/FP_'+imgName2+'.txt'
f = open(path2, "w")
for featurePointStr in featurePointList2:
    f.write(featurePointStr + "\n")
f.close()
cv2.imwrite("./FPImg/"+imgName1+".png", img1)
cv2.imwrite("./FPImg/"+imgName2+".png", img2)
