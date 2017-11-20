
from segmentation import segmentation
from FpEnhancer import *
from Utils import *
from CV_MnExtract import *


# todo : fix file name hard code

################################## Segment ###############################
img = cv2.imread("assets/102_3.tif", cv2.IMREAD_GRAYSCALE)

# # img = cv2.normalize(img,img)
out= segmentation(img, 120) #110 for FP_DB and 45 for FP_DB2

cv2.imshow("segmentation", out)
cv2.imwrite("segmented/1_1.BMP", out)

#
# # initialize the list of threshold methods
# methods = [
# 	("THRESH_BINARY", cv2.THRESH_BINARY),
# 	("THRESH_BINARY_INV", cv2.THRESH_BINARY_INV),
# 	("THRESH_TRUNC", cv2.THRESH_TRUNC),
# 	("THRESH_TOZERO", cv2.THRESH_TOZERO),
# 	("THRESH_TOZERO_INV", cv2.THRESH_TOZERO_INV)]
#
# # loop over the threshold methods
# for (threshName, threshMethod) in methods:
# 	# threshold the image and show it
# 	(T, thresh) = cv2.threshold( img , 110 , 255, threshMethod)
# 	cv2.imshow(threshName, thresh)
# 	cv2.waitKey(0)
# ################################ FpEnhancer ###############################
#
sourceImage = "segmented/1_1.BMP"
# np.set_printoptions(
#     threshold=np.inf,
#     precision=4,
#     suppress=True)

print("Reading image")
image = ndimage.imread(sourceImage, mode="L").astype("float64")
utils.showImage(image, "original", vmax=255.0)

print("Normalizing")
image = utils.normalize(image)
utils.showImage(image, "normalized")

print("Finding mask")
mask = utils.findMask(image)

print("Applying local normalization")
image = np.where(mask == 1.0, utils.localNormalize(image), image)
utils.showImage(image, "locally normalized")

print("Estimating orientations")
orientations = np.where(mask == 1.0, utils.estimateOrientations(image), -1.0)
utils.showOrientations(image, orientations, "orientations", 8)

print("Estimating frequencies")
frequencies = np.where(mask == 1.0, utils.estimateFrequencies(image, orientations), -1.0)

print("Filtering")

image = gaborFilter(image, orientations, frequencies)
image = np.where(mask == 1.0, image, 1.0)
# if options.images > 0:
utils.showImage(image, "gabor")

print("Binarizing")
image = np.where(mask == 1.0, utils.binarize(image, 16), 1.0)
utils.showImage(image, "binarized")

destinationImage = "output.BMP"
# save result image
misc.imsave(destinationImage, image)

# reread the image in cv2 format
img = cv2.imread('output.BMP', 0)
skel = skeletonize(img)
neg = inverse(skel)
cv2.imshow("skeleton", neg)
# show all image during all processes
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

################################ MnExtract ###############################


