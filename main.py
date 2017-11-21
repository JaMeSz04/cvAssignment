
from segmentation import segmentation
from FpEnhancer import *
from Utils import *
from CV_MnExtract import *
import MnMatcher
import os

# todo : fix file name hard code
# todo: show matching

DECIDE_TRESHOLD = 30

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images
images = load_images_from_folder("assets/")

def match(img, img2):

    ################################## Segment ###############################
    #img = cv2.imread("assets/102_3.tif", cv2.IMREAD_GRAYSCALE)

    # img = cv2.normalize(img,img)
    out= segmentation(img, 120) #110 for FP_DB and 45 for FP_DB2
    out2 = segmentation(img2, 120)
    # cv2.imshow("segmentation", out)
    # cv2.imshow("segmentation2", out2)
    cv2.imwrite("segmented/seg.BMP", out)
    cv2.imwrite("segmented/seg2.BMP", out2)

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
    sourceImage = "segmented/seg.BMP"
    sourceImage2 = "segmented/seg2.BMP"
    # np.set_printoptions(
    #     threshold=np.inf,
    #     precision=4,
    #     suppress=True)

    print("Reading image")
    image = ndimage.imread(sourceImage, mode="L").astype("float64")
    image2 = ndimage.imread(sourceImage2, mode="L").astype("float64")
    # utils.showImage(image, "original", vmax=255.0)

    print("Normalizing")
    image = utils.normalize(image)
    image2 = utils.normalize(image2)
    #utils.showImage(image, "normalized")

    print("Finding mask")
    mask = utils.findMask(image)

    print("Applying local normalization")
    image = np.where(mask == 1.0, utils.localNormalize(image), image)
    image2 = np.where(mask == 1.0, utils.localNormalize(image2), image2)
    # utils.showImage(image, "locally normalized")

    print("Estimating orientations")
    orientations = np.where(mask == 1.0, utils.estimateOrientations(image), -1.0)
    # utils.showOrientations(image, orientations, "orientations", 8)
    orientations2 = np.where(mask == 1.0, utils.estimateOrientations(image2), -1.0)
    # utils.showOrientations(image2, orientations2, "orientations2", 8)

    print("Estimating frequencies")
    frequencies = np.where(mask == 1.0, utils.estimateFrequencies(image, orientations), -1.0)
    frequencies2 = np.where(mask == 1.0, utils.estimateFrequencies(image2, orientations2), -1.0)

    print("Filtering")

    image = gaborFilter(image, orientations, frequencies)
    image = np.where(mask == 1.0, image, 1.0)
    image2 = gaborFilter(image2, orientations2, frequencies2)
    image2 = np.where(mask == 1.0, image2, 1.0)
    # if options.images > 0:
    # utils.showImage(image, "gabor")

    print("Binarizing")
    image = np.where(mask == 1.0, utils.binarize(image, 16), 1.0)
    image2 = np.where(mask == 1.0, utils.binarize(image2, 16), 1.0)
    # utils.showImage(image, "binarized")

    destinationImage = "output.BMP"
    destinationImage2 = "output2.BMP"
    # save result image
    misc.imsave(destinationImage, image)
    misc.imsave(destinationImage2, image2)

    # reread the image in cv2 format
    img = cv2.imread('output.BMP', 0)
    img2 = cv2.imread('output2.BMP', 0)
    skel = skeletonize(img)
    skel2 = skeletonize(img2)
    neg = inverse(skel)
    # show all image during all processes
    neg2 = inverse(skel2)
    ################################ MnExtract ###############################
    mn1,mn2 = extractBoom(neg)
    mn3,mn4 = extractBoom(neg2)
    kp1 = MnMatcher.gatherKeyPoints(mn1)
    kp2 = MnMatcher.gatherKeyPoints(mn2)

    kp3 = MnMatcher.gatherKeyPoints(mn3)
    kp4 = MnMatcher.gatherKeyPoints(mn4)

    data = MnMatcher.checkORB(kp1,kp3,neg,neg2)
    data2 = MnMatcher.checkORB(kp2,kp4,neg,neg2)
    print(data)
    print(data2)
    result =  (data + data2) / 2
    print( result > DECIDE_TRESHOLD )

    #plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("result "+str(result))
    return result





match(images[0],images[1])

