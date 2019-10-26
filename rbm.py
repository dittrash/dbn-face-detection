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
imgs = imread_collection('images/new/*.gif')
print("Imported", len(imgs), "images")
print("The first one is",len(imgs[0]), "pixels tall, and",
     len(imgs[0][0]), "pixels wide")
imgs = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgs]
imgsarr = [x.flatten('C') for x in imgs]
imgtest = imread_collection('images/new/neg1.gif')
imgtest = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgtest]
imgtest = [x.flatten('C') for x in imgtest]
#plt.show()
#RBM
#Create a target variable: 1 through 15 for each of the 15 subjects
Y = []
for i in range(5):
    Y.append(0)
for i in range(5):
    Y.append(1)
print(Y)
#Define the RBM, used for feature generation
rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=.01, n_iter=100, n_components=5005)

#Define the Classifier - Logistic Regression will be used in this case
#svm = svm.SVC(C=6000, cache_size=200, class_weight=None, coef0=0.0,
 #           decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  #          max_iter=10000, probability=False, random_state=None, shrinking=True,
    #        tol=0.001, verbose=False)
logistic = LogisticRegression(solver='lbfgs', max_iter=12000,
                              C=6000, multi_class='multinomial', verbose=1)

#dbn = DBN([X_train.shape[1], 300, 10],
 #           learn_rates=0.3,
  #          learn_rate_decays=0.9,
   #         epochs=10,
    #        verbose=1)
#dbn = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
 #                                        learning_rate_rbm=0.05,
  #                                       learning_rate=0.1,
   #                                      n_epochs_rbm=10,
    #                                     n_iter_backprop=100,
     #                                    batch_size=32,
      #                                   activation_function='relu',
       #                                  dropout_p=0.2)
#Combine the two into a Pipeline
#models.append(Pipeline(steps=[('rbm', rbm), ('logistic', logistic)]))
model = Pipeline(steps=[('rbm', rbm), ('rbm1', clone(rbm)),('rbm2', clone(rbm)),('logistic', logistic)])

# Training RBM-Logistic Pipeline
model.fit(imgsarr, Y)
Y_pred = model.predict(imgsarr)
print("Logistic regression using RBM features:\n%s\n" % (metrics.classification_report(Y, Y_pred)))
filename = 'finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))
    
#Selected features for closer examination

first = 8
second = 45

#plt.figure(figsize=(15, 10))
#plt.subplot(121)
#plt.imshow(rbm.components_[first].reshape((77, 65)), cmap=plt.cm.gray_r,
#           interpolation='nearest')
#plt.xticks(())
#plt.yticks(())
#plt.subplot(122)
#plt.imshow(rbm.components_[second].reshape((77, 65)), cmap=plt.cm.gray_r,
#           interpolation='nearest')
#plt.xticks(())
#plt.yticks(())
#plt.show()