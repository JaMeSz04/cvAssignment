
# coding: utf-8

# In[1]:

import cv2
import numpy as np
from tqdm import tqdm


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

len(mnPoints1),len(mnPoints3),len(mnPoints4)


# In[8]:

rgbImg = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)


# In[9]:

for coor in mnPoints1:
    for x in range (-5,5):
        rgbImg[coor[0]+x,coor[1]] = (0,0,255) 
    for y in range (-5,5):
        rgbImg[coor[0],coor[1]+y] = (0,0,255)
        
for coor in mnPoints3:
    for x in range (-5,5):
        rgbImg[coor[0]+x,coor[1]] = (255,0,0) 
    for y in range (-5,5):
        rgbImg[coor[0],coor[1]+y] = (255,0,0)
    


# In[10]:

cv2.imwrite("./minute_example/Input.jpg",rbgImg)

cv2.imshow('imgCross',rgbImg)
cv2.waitKey()

