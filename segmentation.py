import cv2
import numpy as np

def segmentation(img, th):
    retval, segmentedImg = cv2.threshold(img, th, 255, cv2.THRESH_BINARY) #45,255 for FP_DB2// 110,255 for FP_DB
#   cv2.imwrite("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/th/"+ str(i) + "_" + str(j) + ".BMP", segmentedImg)

    row,col = segmentedImg.shape
    lb,tb,rb,bb = False,False,False,False


   # from left border
    if (not lb):
        for r in (range(row)):
            for c in range(col-10):
                if(segmentedImg[r,c]==255 and segmentedImg[r,c+8]==255):
                    break
                else:
                    segmentedImg[r,c] =255
        

    #from right border
    if (not rb):
        for r in range(row-1,0,-1):
            for c in range(col-1,0,-1):
                if(segmentedImg[r,c]==255 and segmentedImg[r,c-10]==255):
                    break
                else:
                    segmentedImg[r,c] =255


    #from top border
    if (not tb):
        for c in range(col):
            for r in range(row):
                if(segmentedImg[r,c]==255):
                    break
                else:
                    segmentedImg[r,c] =255


    #from botton border
    if(not bb):
        for c in range(col-1,0,-1):
            for r in range(row-1,0,-1):
                if(segmentedImg[r,c]==255):
                    break
                else:
                    segmentedImg[r,c] =255

    return segmentedImg

#       out = segment(segmentedImg, 16, 40)

#       cv2.imshow('Output',segmentedImg)
#       cv2.waitKey()


for i in range(1,9):
    for j in range(1,5):
        img = cv2.imread("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/FP_DB/" + str(i) + "_" + str(j) + ".BMP",cv2.IMREAD_GRAYSCALE)
#        img = cv2.imread("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/FP_DB2/101_1.tif",cv2.IMREAD_GRAYSCALE)
        #cv2.imshow("input image",img)
        out = segmentation(img, 110) #110 for FP_DB and 45 for FP_DB2
        cv2.imwrite("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/segmented/"+ str(i) + "_" + str(j) + ".BMP",out)

