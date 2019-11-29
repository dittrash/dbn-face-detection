#image classifier training
#dito15
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.base import clone
from skimage.transform import resize
from skimage.io import imread_collection, imshow
import pickle
#from sklearn import svm
#from nolearn.dbn import DBN
#from dbn.tensorflow.models import SupervisedDBNClassification

#restore the old model
with open('finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)

#load dataset
imgs = imread_collection('images/train/*.gif')
print("Imported", len(imgs), "images")
print("The first one is",len(imgs[0]), "pixels tall, and",
     len(imgs[0][0]), "pixels wide")
imgs = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgs]
imgsarr = [x.flatten('C') for x in imgs]
imgtest = imread_collection('images/new/neg1.gif')
imgtest = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgtest]
imgtest = [x.flatten('C') for x in imgtest]
#plt.show()

#labeling
Y = []
for i in range(50):
    Y.append(0)
for i in range(50):
    Y.append(1)
print(Y)

# Training
model.fit(imgsarr, Y)
Y_pred = model.predict(imgsarr)
print("Logistic regression using RBM features:\n%s\n" % (metrics.classification_report(Y, Y_pred)))
filename = "trained_model.pkl"
pickle.dump(model, open(filename, 'wb'))