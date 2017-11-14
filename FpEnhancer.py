#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
from optparse import OptionParser
import scipy.misc as misc
import scipy.ndimage as ndimage

import Utils as utils

# sourceImage = "./data/1_1.BMP"
# sourceImage = "./data/test2.png"
sourceImage = "DeOutput.BMP"


def gaborKernel(size, angle, frequency):
    """
    Create a Gabor kernel given a size, angle and frequency.

    refer to https://github.com/rtshadow/biometrics.git
    """

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


def gaborFilterSubdivide(image, orientations, frequencies, rect=None):
    if rect:
        y, x, h, w = rect
    else:
        y, x = 0, 0
        h, w = image.shape

    result = np.empty((h, w))

    orientation, deviation = utils.averageOrientation(
        orientations[y:y + h, x:x + w], deviation=True)

    if (deviation < 0.2 and h < 50 and w < 50) or h < 6 or w < 6:
        # print(deviation)
        # print(rect)

        frequency = utils.averageFrequency(frequencies[y:y + h, x:x + w])

        if frequency < 0.0:
            result = image[y:y + h, x:x + w]
        else:
            kernel = gaborKernel(16, orientation, frequency)
            result = utils.convolve(image, kernel, (y, x), (h, w))

    else:
        if h > w:
            hh = h // 2

            result[0:hh, 0:w] = \
                gaborFilterSubdivide(image, orientations, frequencies, (y, x, hh, w))

            result[hh:h, 0:w] = \
                gaborFilterSubdivide(image, orientations, frequencies, (y + hh, x, h - hh, w))
        else:
            hw = w // 2

            result[0:h, 0:hw] = \
                gaborFilterSubdivide(image, orientations, frequencies, (y, x, h, hw))

            result[0:h, hw:w] = \
                gaborFilterSubdivide(image, orientations, frequencies, (y, x + hw, h, w - hw))

    if w > 20 and h > 20:
        result = utils.normalize(result)

    return result


if __name__ == '__main__':
    np.set_printoptions(
        threshold=np.inf,
        precision=4,
        suppress=True)

    print("Reading image")
    image = ndimage.imread(sourceImage, mode="L").astype("float64")
    # if options.images > 0:
    utils.showImage(image, "original", vmax=255.0)

    print("Normalizing")
    image = utils.normalize(image)
    # if options.images > 1:
    utils.showImage(image, "normalized")

    print("Finding mask")
    mask = utils.findMask(image)
    # if options.images > 1:
    #utils.showImage(mask, "mask")

    print("Applying local normalization")
    image = np.where(mask == 1.0, utils.localNormalize(image), image)
    # if options.images > 1:
    utils.showImage(image, "locally normalized")

    print("Estimating orientations")
    orientations = np.where(mask == 1.0, utils.estimateOrientations(image), -1.0)
    # if options.images > 0:
    utils.showOrientations(image, orientations, "orientations", 8)

    print("Estimating frequencies")
    frequencies = np.where(mask == 1.0, utils.estimateFrequencies(image, orientations), -1.0)
    # if options.images > 1:
    #utils.showImage(utils.normalize(frequencies), "frequencies")

    print("Filtering")
    # if options.subdivide:
    #     image = utils.normalize(gaborFilterSubdivide(image, orientations, frequencies))
    # else:
    image = gaborFilter(image, orientations, frequencies)
    image = np.where(mask == 1.0, image, 1.0)
    # if options.images > 0:
    utils.showImage(image, "gabor")

    # if options.binarize:
    print("Binarizing")
    image = np.where(mask == 1.0, utils.binarize(image,8), 1.0)
    # if options.images > 0:
    utils.showImage(image, "binarized")

    # if options.images > 0:
    plt.show()

    destinationImage = "output.BMP"
    misc.imsave(destinationImage, image)

    img = cv2.imread('output.BMP', 0)
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

    cv2.imshow("skel", skel)
    cv2.waitKey()
    cv2.destroyAllWindows()


