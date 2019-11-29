import glob
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pickle
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import io

with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)
im = cv2.imread("images/test/testwide.jpg")
counter = 0
#im = cv2.cvtColor(np.array(Image.open(images[0])), cv2.COLOR_BGR2GRAY)
# selective search
#im = cv2.resize(im, (newWidth, newHeight))    
# create Selective Search Segmentation Object using default parameters
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# set input image on which we will run segmentation
ss.setBaseImage(im)

# Switch to fast but low recall Selective Search method
#if (sys.argv[2] == 'f'):
ss.switchToSelectiveSearchFast()

# Switch to high recall but slow Selective Search method
#elif (sys.argv[2] == 'q'):
#ss.switchToSelectiveSearchQuality()
# if argument is neither f nor q print help message


# run selective search segmentation on input image
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))
 
# number of region proposals to show
#numShowRects = 100
# increment to increase/decrease total number
# of reason proposals to be shown

imOut = im.copy()

# itereate over all the region proposals
for i, rect in enumerate(rects):
    x, y, w, h = rect
    timage = imOut[y:y+h,x:x+w]
    wintest = cv2.resize(timage, (231,195), interpolation = cv2.INTER_AREA)
    wintest = rgb2gray(wintest)
    wintest = resize(wintest,(77,65),mode='constant', anti_aliasing=False)
    wintest = wintest.flatten('C')
    wintest = wintest.reshape(1,-1)
    #predict = model.predict(wintest)
    prob =  model.predict_proba(wintest)[:,1]
    prob = float(prob)
    if prob >= 0.90:
        cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        print("{0:.2f}".format(prob),"<-face")
        counter += 1
    else:
        print("{0:.2f}".format(prob),"<-not face")
    # daw rectangle for region proposal till numShowRects
cv2.imshow("Output", imOut)
print(counter)
cv2.waitKey(0)

#cv2.destroyAllWindows()