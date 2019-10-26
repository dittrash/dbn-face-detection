#image classifier testing
#dito15
import pickle
from skimage.io import imread_collection, imshow
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import io
imgtest = imread_collection('images/neg12.gif')
imgtest = [rgb2gray(x) for x in imgtest]
imgtest = [resize(x,(77,65),mode='constant', anti_aliasing=False) for x in imgtest]
imgtest = [x.flatten('C') for x in imgtest]
with open('finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)
print('class predicted:', model.predict(imgtest))