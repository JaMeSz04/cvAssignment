import cv2


################ begin segmentation #####################

def segmentation(img, th):
    # retval, img = cv2.threshold(img, th, 255, cv2.THRESH_OTSU)
    retval, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)

    # row,col = img.shape
    # dsum = 0
    # avg = []
    # for i in range(row):
    #     for j in range(col):
    #         dsum += img[i, j]
    #         if (img[i,j] % BLOCK_LENGTH == 0):
    #             avg.append(dsum / BLOCK_LENGTH)
    #             dsum = 0
    #
    # newSum = sum(avg) / len(avg)
    # print(newSum)
    # retval, segmentedImg = cv2.threshold(img, newSum, 255, cv2.THRESH_BINARY)
    # return segmentedImg

    row, col = img.shape
    BLOCK_LENGTH = int(row / 8)
    print(img.shape)
    dsum = 0
    avg = []
    count = 0
    count2 = 0
    # blockSize = 16
    # theshold = 25
    # blockCol = int(math.floor(col / blockSize))
    # blockRow = int(math.floor(row / blockSize))
    # for colIndex in range(0, blockSize):
    #     for rowIndex in range(0, blockSize):
    #         sum = 0
    #         for i in range(colIndex * blockCol, colIndex * blockCol + blockCol):
    #             for j in range(rowIndex * blockRow, rowIndex * blockRow + blockRow):
    #                 sum = sum + img[i, j]
    #         blockAvg = sum / (blockCol * blockRow)
    #         print(blockAvg)
    #         if blockAvg < theshold:
    #             for i in range(colIndex * blockCol, colIndex * blockCol + blockCol):
    #                 for j in range(rowIndex * blockRow, rowIndex * blockRow + blockRow):
    #                     img[i, j] = 255;

    for i in range(0, row):
        count = 0
        for j in range(0, col // 2):
            if img[i, j] == 255:
                count += 1
                if count > 5:
                    break
            else:
                img[i, j] = 255

        count = 0
        for j in range(col - 1, col // 2, -1):
            if img[i, j] == 255:
                count += 1
                if count > 5:
                    break
            else:
                img[i, j] = 255








                # while(1):
    # for i in range(count,count+BLOCK_LENGTH):
    #         for j in range(count2,count2+BLOCK_LENGTH):
    #             dsum += img[i, j]
    #         # completed all pixels in the block
    #         av = dsum / BLOCK_LENGTH
    #         if(av <= 40):
    #             # replace whole block with white color
    #             img[count:count+BLOCK_LENGTH,count2:count2+BLOCK_LENGTH] = 255
    #         # reset sum
    #         dsum = 0
    #
    #         # increment row count
    #         count = count + BLOCK_LENGTH
    #         print("count " + str(count))
    #         # prevent index out of bound
    #         if ( count + BLOCK_LENGTH >= col):
    #             break
    #
    #     # increment col count
    #     count2 = count2 + BLOCK_LENGTH
    #     print("count2 " + str(count2))
    #     # prevent index out of bound
    #     if (count2 + BLOCK_LENGTH >= row):
    #         # reset count2
    #         break






    return img


'''
for i in range(1,9):
    for j in range(1,5):
        img = cv2.imread("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/FP_DB/" + str(i) + "_" + str(j) + ".BMP",cv2.IMREAD_GRAYSCALE)
        out = segmentation(img, 110) #110 for FP_DB and 45 for FP_DB2
        cv2.imwrite("/Users/wiranchana/Desktop/kmitl/S1Y4/cv/segmented/"+ str(i) + "_" + str(j) + ".BMP",out)
'''

'''
img = cv2.imread("assets/1_1.BMP", cv2.IMREAD_GRAYSCALE)
out = segmentation(img, 110) #110 for FP_DB and 45 for FP_DB2
cv2.imwrite("segmented/1_1.BMP", out)
show = cv2.imread("segmented/1_1.BMP", cv2.IMREAD_GRAYSCALE)
cv2.imshow("segmentation", show)
cv2.waitKey()
cv2.destroyAllWindows()
'''

####### end segmentation ############################################
