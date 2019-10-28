#image classifier testing
#dito15
import pickle
from skimage.io import imread_collection, imshow, imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import io
imgtest = imread('images/test/test1.gif')
imgtest = rgb2gray(imgtest)
imgtest = resize(imgtest,(77,65),mode='constant', anti_aliasing=False)
imgtest = imgtest.flatten('C')
imgtest = imgtest.reshape(1,-1)
print(imgtest)

with open('finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)
#class prediction
print('class predicted:', model.predict(imgtest))
#probability
prob =  model.predict_proba(imgtest)[:,1]
prob = float(prob)
print("{0:.2f}".format(prob))