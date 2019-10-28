# import the necessary packages
from pyimage.helpers import pyramid
from pyimage.helpers import sliding_window
import argparse
import time
import cv2
import pickle
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import io
from skimage.io import imread_collection, imshow, imread

with open('finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)

# construct the argument parser and parse the arguments

# load the image and define the window width and height
image = imread('images/test/testwide.jpg')
(winW, winH) = (195, 231)
total = 0
Y = [1]
# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        wintest = window
        wintest = rgb2gray(wintest)
        wintest = resize(wintest,(77,65),mode='constant', anti_aliasing=False)
        wintest = wintest.flatten('C')
        wintest = wintest.reshape(1,-1)
        predict = model.predict(wintest)
        #using class prediction
        if predict == 1:
            cv2.waitKey(0)
        print(predict)
        #using class probability
        #prob =  model.predict_proba(wintest)[:,1]
        #prob = float(prob)
        #print("{0:.2f}".format(prob))
        #if prob >= 0.74:
        #    cv2.waitKey(0)
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Detector", clone)
        cv2.waitKey(1)
        time.sleep(0.025)
#print('face detected:',total)