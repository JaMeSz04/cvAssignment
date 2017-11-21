import cv2
import numpy as np
import main

#-----------------------------
def genFpImgName(fpNo, imgNo, ext):
    return fpNo+ "_" + imgNo + ext
        
#-----------------------------
class FpMatchingEvaluator:
    def __init__(self):
        self.outFileName = "matching_result.txt"
        self.outFile = open(self.outFileName, "w")
        self.similaritySameFinger = []
        self.similarityDiffFinger = []
        self.falseAccept = [0] * 101
        self.falseReject = [0] * 101
        
    def evaluate(self, dbFileName, fpMatcher):
        #reset attributes
        self.similaritySameFinger = []
        self.similarityDiffFinger = []
        #open file stream
        inFile = open(dbFileName, "r")
        #get file extension
        ext = inFile.readline().rstrip()
        #get path
        path = inFile.readline().rstrip()
        #get all image names
        line_list = inFile.readlines()
        fpFiles = [x.split() for x in line_list]
        #for all pairs of fingerprint images
        for i in range(len(fpFiles)):
            for j in range(i + 1, len(fpFiles)):
                #read images
                name1 = genFpImgName(fpFiles[i][0], fpFiles[i][1], ext)
                print(path + name1)
                fpImg1 = cv2.imread(path + name1, cv2.IMREAD_GRAYSCALE)
                name2 = genFpImgName(fpFiles[j][0], fpFiles[j][1], ext)
                fpImg2 = cv2.imread(path + name2, cv2.IMREAD_GRAYSCALE)
                print(path + name2)
                #matching
                s = fpMatcher.match(fpImg1, fpImg2)
                print(name1, "vs", name2, ":", s)
                self.outFile.write(name1 + " vs " + name2 + " : " + str(s))

                #save similarity
                if fpFiles[i][0] == fpFiles[j][0]:
                    self.similaritySameFinger.append(s)
                else:
                    self.similarityDiffFinger.append(s)

        #calculate the performace
        self.calculateResults()
                    
    def calculateResults(self):
        for th in range(101):
            numFalseAccept = 0
            numFalseReject = 0

            #check genuine attempt vector
            for s in self.similaritySameFinger:
                if s < th:
                    numFalseReject += 1
                    
            #check imposter attempt vector
            for s in self.similarityDiffFinger:
                if s >= th:
                    numFalseAccept += 1

            #compute false acceptance and false rejection rates
            self.falseAccept[th] = numFalseAccept / len(self.similarityDiffFinger) * 100
            self.falseReject[th] = numFalseReject / len(self.similaritySameFinger) * 100

            #display
            print("threshold = ", th, ":", \
                  "false accept = ", self.falseAccept[th], \
                  "false reject = ", self.falseReject[th])
            self.outFile.write("threshold = " + str(th) + ":" + \
                  "false accept = " + str(self.falseAccept[th]) + \
                  "false reject = " + str(self.falseReject[th]))

        eer = self.calculateEER()
        print("EER = ", eer)
        self.outFile.write("EER = " + str(eer))
            
    def calculateEER(self):
        t1 = t2 = 0
        #find EER_low
        for th in range(101):
            if self.falseReject[th] > self.falseAccept[th]:
                t1 = th
                break

        #find EER_high
        for th in range(101):
            if self.falseReject[th] < self.falseAccept[th]:
                t2 = th
                break

        #cal EER
        if (self.falseReject[t1] + self.falseAccept[t1]) < \
           (self.falseReject[t2] + self.falseAccept[t2]):
            eer_low, eer_high = self.falseReject[t1], self.falseAccept[t1]
        else:
            eer_low, eer_high = self.falseAccept[t2], self.falseReject[t2]
        
        return (eer_low + eer_high) / 2

#-----------------------------

if __name__ == "__main__":
    fpEvl = FpMatchingEvaluator()
    fpEvl.evaluate("./ShubU/DB_Info.txt", main)

#-----------------------------
