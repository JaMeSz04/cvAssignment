import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


EQUAL_TRESHOLD = 62.9

def translate( coor, delta ):
    return ( coor[0] + delta[0] , coor[1] + delta[1] )

#input array of tuple
def isMatch(inp1, inp2):
    input1 = inp1
    input2 = inp2
    for i in range(1, len(input1)):
        pair1 = (input1[i-1], input1[i]) #choose a pair of minutiae
        pair2 = (input2[i-1], input2[i])
        translationDelta = (pair1[0][0] - pair2[0][0], pair1[0][1] - pair2[0][1])
        rotationDelta = math.degrees(math.atan2( pair2[1][1] - pair1[1][1], pair2[1][0] - pair1[1][0] ))
        input2[i-1], input2[i] = translate( pair2 , translationDelta )



def gatherKeyPoints( vectors ):
    return [ cv2.KeyPoint(x[0], x[1], 31) for x in vectors]

def checkORB(keyPoint1, keyPoint2, img, img2):

    out = np.zeros(shape = (len(img), len(img)))
    orb = cv2.ORB_create()

    out2 = np.zeros(shape=(len(img2), len(img2)))
    orb2 = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # find the keypoints with ORB
    kp2 = orb.detect(img2, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, keyPoint1)
    # compute the descriptors with ORB
    kp2, des2 = orb.compute(img2, keyPoint2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    #print(matches)

    img3 = cv2.drawMatches(img, kp, img2, kp2, matches[:30], out, flags=2)

    #plt.imshow(img3), plt.show()
    return evaluate(keyPoint1, matches, 50)

def evaluate(inp1, matches, percentage):
    print((len(matches) / len(inp1)) * 100)
    if (len(inp1) < len(matches)):
        print("Something must be wrong!!!")
        return
    if ((len(matches) / len(inp1)) * 100 > percentage):
        return (len(matches) / len(inp1)) * 100
    return (len(matches) / len(inp1)) * 100

