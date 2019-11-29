import glob
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

im = cv2.imread("images/test/test2.jpg")
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
numShowRects = 1000
# increment to increase/decrease total number
# of reason proposals to be shown

imOut = im.copy()

# itereate over all the region proposals
for i, rect in enumerate(rects):
    # daw rectangle for region proposal till numShowRects
    if (i < numShowRects):
        x, y, w, h = rect
        cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    else:
        break

# show output
cv2.imshow("Output", imOut)

# record key press
cv2.waitKey(0)

#cv2.destroyAllWindows()