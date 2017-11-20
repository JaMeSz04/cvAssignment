
# coding: utf-8

# In[1]:

import cv2
import numpy as np
from tqdm import tqdm
import math


# In[2]:

img = cv2.imread('./minute_example/Input.jpg',cv2.IMREAD_GRAYSCALE)


# In[3]:


ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,value = 50)


# In[4]:

row,col = img.shape


# In[5]:

offset = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


# In[6]:

crossingCount = 0
mnPoints1 = []
mnPoints3 = []
mnPoints4 = []
for x in tqdm(range (row)):
    for y in range (col):
        if(img[x,y]==0):
            values = []
            for offsetX, offsetY in offset:
                values.append(img[x+offsetX,y+offsetY])   
            for nb in range(0,8):
                if(int(values[nb]) + int(values[nb+1])==255 ):
                    crossingCount +=1
            crossingCount /= 2
            if (crossingCount ==3 ):
                mnPoints3.append((x,y))
            elif (crossingCount ==1 ):
                #print(values)
                mnPoints1.append((x,y))
            elif (crossingCount ==4 ):
                mnPoints4.append((x,y))
           # print crossingCount
            crossingCount = 0    


# In[7]:

mnPoints = mnPoints1+mnPoints3


# In[8]:

rgbImg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)


# In[9]:

someMn = []
for coor in mnPoints:
    x = coor[0]
    y = coor[1]
    cMn1,cMn2,cMn3,cMn4 = False,False,False,False
    
    for xinter in range(x+1,row-1):
        if(img[xinter,y]==0):
            cMn1=True
    for xinter in range(x-1,0,-1):
        if(img[xinter,y]==0):
            cMn2=True
    
    for yinter in range(y+1,col-1):
        if(img[x,yinter]==0):
            cMn3=True
    for yinter in range(y-1,0,-1):
        if(img[x,yinter]==0):
            cMn4=True
    if(cMn1 and cMn2 and cMn3 and cMn4):
        someMn.append((x,y))
        


# In[10]:

for coor in someMn:
    for x in range (-5,5):
        rgbImg[coor[0]+x,coor[1]] = (0,0,255) 
    for y in range (-5,5):
        rgbImg[coor[0],coor[1]+y] = (0,0,255)


# In[11]:

allPoints = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)


# In[12]:

for coor in mnPoints1:
    for x in range (-5,5):
        allPoints[coor[0]+x,coor[1]] = (0,0,255) 
    for y in range (-5,5):
        allPoints[coor[0],coor[1]+y] = (0,0,255)
for coor in mnPoints3:
    for x in range (-5,5):
        allPoints[coor[0]+x,coor[1]] = (255,0,0) 
    for y in range (-5,5):
        allPoints[coor[0],coor[1]+y] = (255,0,0)   


# In[13]:

cv2.imwrite("./minute_example/Output.jpg",rgbImg)


# In[ ]:

cv2.imshow('imgCross',rgbImg)
cv2.imshow('allCross',allPoints)
cv2.imshow('original',img)
cv2.waitKey()


# In[ ]:



