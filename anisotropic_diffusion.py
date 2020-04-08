# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:11:36 2016

@author: Vish
"""




# Name : Vishrut Krashak                ID : 12085766



# Name : Stephen Geary                  ID : 10060944






import numpy as np 
from PIL import Image
import matplotlib.pylab as pyp
import matplotlib.pyplot as plt
import pylab


#----------------------------specifying the name of the image to be used for testing------------------------
# since the image is in the same folder as the project, we provide it with the name instead of the full path


resultImage = np.array(Image.open('noisy-empire.png').convert('L')) 
#resultImage = np.array(Image.open('noisy-empire.png').convert('L')) 
"""

Change value of C before checking the result with noisy rectangle

""" 



#-------------------------------finding the value of the minimum and the maximum pixel----------------------

im_min, im_max = resultImage.min(), resultImage.max()

print "Original image:", resultImage.shape, resultImage.dtype, im_min, im_max

#------------------------------conversion of the image---------------------------------------
resultImage = (resultImage - im_min) / (float)(im_max - im_min)   
print "Perona-Malik Anisotropic Diffusion:", resultImage.shape, resultImage.dtype, resultImage.min(), resultImage.max()

#----------------------------------displaying the image provided by the lecturer--------------------
pyp.figure('Image before performing anisotropic diffusion')
pyp.imshow(resultImage, cmap='gray')
pyp.axis('on')

#---------------------------------------display image------------------------------------
pyp.show() 
    
def f(delta,b):
    #return 1/(1 + (np.power(delta,2)/np.power(b,2)))
    return np.exp(-1* (np.power(delta,2))/(np.power(b,2)))

def anisotropic_diffusion(im, steps,C=0.99, lam = 0.25):  
    

    im_new = np.zeros(im.shape, dtype=im.dtype) 
    x = list()
    y = list()    
    
    for t in range(steps):
        x.append(t)
        dn = im[:-2,1:-1] - im[1:-1,1:-1] 
        ds = im[2:,1:-1] - im[1:-1,1:-1] 
        de = im[1:-1,2:] - im[1:-1,1:-1] 
        dw = im[1:-1,:-2] - im[1:-1,1:-1] 
        b=make_b(dn, ds, de, dw, C, H_BINS=50)
        y.append(b)
        
        im_new[1:-1,1:-1] = im[1:-1,1:-1] +\
                            lam * (f(dn,b)*dn + f (ds,b)*ds + 
                                    f(de,b)*de + f (dw,b)*dw) 
        im = im_new 
        
    plt.figure("New fig")
    plt.plot(x,y,'bo', color="red")
    plt.vlines(x, [0], y)
    plt.xlabel('Iteration (Cycles)')
    plt.ylabel('b Value')
    plt.show()
        
#----------------------------THIS IS WHERE THE CODE WAS GIVING ME ERRORS----------------------------
        
        #v = (np.sqrt(float(2 * lam * steps)))
    return im# ,v#,np.sqrt(2 * lam * steps)
#---------------------------------------------------------------------------------------------------  


def make_b(dn, ds, de, dw, C, H_BINS=50):
    #“Per-cycle calculation of b for a given C. Call it inside the main diffusion loop.”
    gm = np.fabs((dn, ds, de, dw))
    hist, edges = np.histogram(gm, H_BINS, density=True)
    bin_width = edges[1] - edges[0]
    hist *= bin_width
    acc, b = hist[0], 0.0
    for j in range(1, len(hist)):
        if acc > C: break
        acc, b = acc + hist[j], b + bin_width
    #print ("The value to be used is " + b)
    return b


#im2 = anisotropic_diffusion(resultImage, 40, 0.3, 0.25)  ----- For noisy rectangle, b is higher for a cleaner image
#--------------------------------------------------------------

im2 = anisotropic_diffusion(resultImage, 40,0.9, 0.25)
pyp.figure('Image cleaned by anisotropic diffusion')
pyp.imshow(im2, cmap='gray')
pyp.axis('on')
#---------------------------------------Display the cleaned image---------------------------------
pyp.show() 
