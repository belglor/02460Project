from scipy.misc import imread
#from MCGFunctions import *
import matlab.engine
import scipy.io as sio
import os
import numpy as np
from scipy.misc import imshow

this_dir = '/home/soren/Desktop/MCG/mcg/pre-trained'

os.chdir(this_dir)

img = imread('101087.jpg')

eng = matlab.engine.connect_matlab()
eng.cd(this_dir)
print(eng.pwd())

eng.install_func()
eng.run_im2mcg('demos','101087.jpg','fast') #use fast or accurate
# It takes WAAAAY too long time to parse all masks directly, so we save them to mat file and load them in python

masks = sio.loadmat('matlabOut.mat')['masks']

#show 10 first and 10 last region proposals
#10 first
tmpImg = np.zeros(img.shape)
for i in range(10):
    for k in range(3):
        tmpImg[:,:,k] = img[:,:,k]*masks[:,:,i]
    imshow(tmpImg)
#10 last
for i in range(10):
    for k in range(3):
        tmpImg[:, :, k] = img[:, :, k] * masks[:, :, -i]
    imshow(tmpImg)

#[candidates_scg, ucm2_scg] = im2mcg(img,'fast')