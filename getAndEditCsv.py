import cv2
import glob
import numpy as np
import pandas as pd






def getImagesFromDest(width,height,a):
        resizeRate = (width, height)
        datasetPath = "C:/Users/Sefa/Desktop/odevDB"
        tempImage = []
        tempSetId = []
        images = []
        imageNameSet = ["/aripiprazole","/betamethasone","/cinoxacin","/ebselen","/haloperidol","/irbesartan"]
        resizeRate = (width, height)



        if (a == 0):
                datasetPath = "C:/Users/Sefa/Desktop/tempOdevDB"
                csvRead = pd.read_excel("tempDummyCsv.xlsx")
        else:
                csvRead = pd.read_excel("dummyCsv.xlsx")

        for x in range(len(imageNameSet)):
                entirePath = datasetPath + imageNameSet[x] + "/*.png"
                images.append([cv2.resize(cv2.imread(file),resizeRate) for file in glob.glob(entirePath)])

        images = images[0] + images[1] + images[2] + images[3] + images[4] + images[5]

        return np.array(csvRead[0]),np.array(images)







