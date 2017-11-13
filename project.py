
import cv2
import numpy as np

img = cv2.imread("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/FP_DB/1_1.BMP",cv2.IMREAD_GRAYSCALE)
#img = cv2.imread("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/FP_DB2/101_1.tif",cv2.IMREAD_GRAYSCALE)
cv2.imshow("input image",img)

retval, segmentedImg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#cv2.imshow('seg',segmentedImg)

row,col = segmentedImg.shape
lb,tb,rb,bb = False,False,False,False


#check if left border contain finger
for r in range(row):
    if(segmentedImg[r,0]==255):
        lb = True
        
#check if right border contain finger
for r in range(row):
    if(segmentedImg[r,col-1]==255):
        rb = True

#check if top border contain finger
for c in range(col):
    if(segmentedImg[0,c]==255):
        tb = True
        
#check if bottom border contain finger
for c in range(col):
    if(segmentedImg[row-1,c]==255):
        bb = True
        

# from left border
if (not lb):
    for r in (range(row)):
        for c in range(col):
            if(segmentedImg[r,c]==255):
                break
            else:
                segmentedImg[r,c] =255
    

#from right border
if (not rb):
    for r in range(row-1,0,-1):
        for c in range(col-1,0,-1):
            if(segmentedImg[r,c]==255):
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

cv2.imshow('Output',segmentedImg)
cv2.waitKey()
