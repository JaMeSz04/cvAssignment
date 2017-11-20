import cv2
import numpy as np

def segmentation(img, th):
    retval, segmentedImg = cv2.threshold(img, th, 255, cv2.THRESH_BINARY)

    row,col = segmentedImg.shape

# from left border

    for r in (range(row)):
        for c in range(col-10):
            if(segmentedImg[r,c]==255 and segmentedImg[r,c+8]==255):
                break
            else:
                segmentedImg[r,c] =255

#from right border

    for r in range(row-1,0,-1):
        for c in range(col-1,0,-1):
            if(segmentedImg[r,c]==255 and segmentedImg[r,c- 10]==255):
                break
            else:
                segmentedImg[r,c] =255

#from top border

    for c in range(col):
        for r in range(row):
            if(segmentedImg[r,c]==255):
                break
            else:
                segmentedImg[r,c] =255

#from botton border

    for c in range(col-1,0,-1):
        for r in range(row-1,0,-1):
            if(segmentedImg[r,c]==255):
                break
            else:
                segmentedImg[r,c] =255

    return segmentedImg

for i in range(1,9):
    for j in range(1,5):
        img = cv2.imread("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/FP_DB/" + str(i) + "_" + str(j) + ".BMP",cv2.IMREAD_GRAYSCALE)
        out = segmentation(img, 110) #110 for FP_DB and 45 for FP_DB2
        cv2.imwrite("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/segmented/"+ str(i) + "_" + str(j) + ".BMP",out)
