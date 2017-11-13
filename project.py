
import cv2
import numpy as np

def distanceThresholding(colorImg, color, T):
    rows, cols ,chs= colorImg.shape
    outImg = np.zeros((rows, cols), np.uint8)
    outColorImg = np.zeros(colorImg.shape, np.uint8)
    for r in range(rows):
        for c in range(cols):
            d = colorImg[r,c,:] - color[:]
            dist = np.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])
            if dist < T:
                outImg[r,c] = 255
                outColorImg[r,c,:] = colorImg[r,c,:]
    return outImg, outColorImg

img = cv2.imread("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/FP_DB/1_1.BMP",cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/FP_DB2/101_1.tif",cv2.IMREAD_GRAYSCALE)
#img = cv2.resize(img, (500, 500)) 
cv2.imshow("input image",img)

retval, segmentedImg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#outImg , outColorImg= distanceThresholding(img,(0,255,0),200)
#print("Otsu's threshold = ", retval)


cv2.imshow('seg',segmentedImg)

invertedImg  = cv2.bitwise_not(segmentedImg)
cv2.imshow('negative',invertedImg)

highpass = np.ones((3,3), np.int8) * -1
highpass[1,1] = 8

outImg = cv2.filter2D(invertedImg, cv2.CV_8U, highpass)
#cv2.imshow('highpass',outImg)

cv2.waitKey()
