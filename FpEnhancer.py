import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndimage

import Utils as utils


## begin: Place a code for image segmentation here

## end: image segmentation part

# sourceImage = "segmented/1_1.BMP"

#  modified from https://github.com/rtshadow/biometrics.git and https://github.com/tommythorsen/fingerprints
## begin: Fingerprint enhancement part
def gaborKernel(size, angle, frequency):
    # Create a Gabor kernel given a size, angle and frequency.
    angle += np.pi * 0.5
    cos = np.cos(angle)
    sin = -np.sin(angle)
    
    yangle = lambda x, y: x * cos + y * sin
    xangle = lambda x, y: -x * sin + y * cos
    
    xsigma = ysigma = 4
    
    return utils.kernelFromFunction(size, lambda x, y:
                                    np.exp(-(
                                             (xangle(x, y) ** 2) / (xsigma ** 2) +
                                             (yangle(x, y) ** 2) / (ysigma ** 2)) / 2) *
                                    np.cos(2 * np.pi * frequency * xangle(x, y)))


def gaborFilter(image, orientations, frequencies, w=16):
    result = np.empty(image.shape)
    
    height, width = image.shape
    for y in range(0, height - w, w):
        for x in range(0, width - w, w):
            orientation = orientations[y + w // 2, x + w // 2]
            frequency = utils.averageFrequency(frequencies[y:y + w, x:x + w])
            
            if frequency < 0.0:
                result[y:y + w, x:x + w] = image[y:y + w, x:x + w]
                continue
            
            kernel = gaborKernel(16, orientation, frequency)
            result[y:y + w, x:x + w] = utils.convolve(image, kernel, (y, x), (w, w))

    return utils.normalize(result)

def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    
    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel

def inverse(img):
    neg = cv2.bitwise_not(img)
    return neg

    

### end: Fingerprint enhancement part




