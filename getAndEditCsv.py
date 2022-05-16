import cv2
import glob
import numpy as np

datasetPath = "C:/Users/Sefa/Desktop/odevDB"
width = 128
height = 128
resizeRate = (width, height)



def getImagesFromDest():
        tempImage = []
        tempSetId = []
        images = []
        imageNameSet = ["/aripiprazole","/betamethasone","/cinoxacin","/ebselen","/haloperidol","/irbesartan"]
        resizeRate = (width, height)



        for x in range(len(imageNameSet)):
                entirePath = datasetPath + imageNameSet[x] + "/*.png"
                images.append([cv2.imread(file) for file in glob.glob(entirePath)])



        cv2.imshow("A", images[0][0])
        cv2.waitKey()


