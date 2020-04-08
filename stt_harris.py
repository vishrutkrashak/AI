# -*- coding: utf-8 -*-
"""
Created on Tue Mar 08 20:55:03 2016

@author: Vish
"""



####################            Vishrut Krashak         12085766            ############################################



##################              Stephen Geary           10060944            ######################################


# -*- coding: utf-8 -*-
from scipy.ndimage import filters
import numpy as np
from PIL import Image
from pylab import *
#import matplotlib.pyplot as plt
#import os, sys




def compute_harris_response(im1, sigma = 2):
    """Compute the Harris corner detector response function 
    for each pixel in a graylevel image. """
    
    
    imx = np.zeros(im1.shape)
    filters.gaussian_filter(im1, (sigma, sigma),(0,1), imx)
    #imy = zeros(im1.shape)
    imy = np.zeros(im1.shape)
    filters.gaussian_filter(im1, (sigma,sigma), (1,0), imy)
    # compute components of the Harris matrix
    A = filters.gaussian_filter(imx*imx,sigma)
    B = filters.gaussian_filter(imx*imy,sigma)
    C = filters.gaussian_filter(imy*imy,sigma)
   
    # determinant and trace
    
    M_det = (A*C) - (B**2)
    M_tr = (A+C)
    #k = 0.6
    #R = M_det + (k*M_tr)
    
    return M_det/M_tr#R
    
 
    
def get_harris_points(harris_im1,min_dist=10,threshold=0.1):
    """ Return corners from a Harris response image 
    min_dist is the minimum number of pixels separating
    corners and image boundary."""
    
    # Find top corner candidates above a threshold
    corner_threshold = harris_im1.max() * threshold
    harris_im1_th = (harris_im1 > corner_threshold) #returns boolean true or false
    
    # Get coordinates of candidates
    coords = np.array(harris_im1_th.nonzero()).T
    
    # ...and their values
    candidate_values = [harris_im1[c[0],c[1]] for c in coords]
    # sort candidates
    indices = np.argsort(candidate_values)
    
    # store allowed point locations in array
    allowed_locations = np.zeros(harris_im1.shape, dtype = 'bool')#non-maximum suppression
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = True
    
    # select the best points taking min_distance into account
    
    filtered_coords = []
    for i in indices[::-1]:# walk the array from back to front
        r,c = coords[i]    # note the reversed indices
        if allowed_locations[r,c]:
            filtered_coords.append((r,c))
            allowed_locations[r-min_dist:r+min_dist+1,c-min_dist:c+min_dist+1] = False
                
    return filtered_coords
    

def plot_harris_points(image,filtered_coords): #Plots Harris corners on graph
    
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],
                [p[0] for p in filtered_coords],'ro')
    axis('off')
    show()

  
    
def get_descriptors(image,filtered_coords,wid=5):
    """For each point return pixel values around the point
    using a neighbourhood of width 2*wid+1. (Assume points are
    extracted with min_distance > wid)."""
    
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0] - wid:coords[0] + wid +1,
                coords[1] - wid:coords[1] + wid +1].flatten()
        desc.append(patch)
        
    return desc
        
def match(desc1,desc2,threshold=0.95):
    """For each corner point descriptor in the first image,
    select its match to second image using
    normalized cross correlation."""
    
    n = len(desc1[0])
    
    #pairwise distances
    d = -ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
                
    ndx = np.argsort(-d)
    matchscores = ndx[:,0]
    
    return matchscores


"""
This function here should remove all the non-symmetric matches 
and return images with symmetric matches,but from the results there 
seems to be a funny match in both the images 

"""
def match_twosided(desc1,desc2,threshold=0.95):
	""" Two-sided symmetric version of match(). """
	matches_12 = match(desc1,desc2,threshold)
	matches_21 = match(desc2,desc1,threshold)
	ndx_12 = where(matches_12 >= 0)[0]

	# remove matches that are not symmetric
	for n in ndx_12:
		if matches_21[matches_12[n]] != n:
			matches_12[n] = -1 

	return matches_12


def appendimages(im1,im2):
	""" Return a new image that appends the 2 images next to each other. """
	
	# select the image with the fewest rows and fill in enough empty rows
	rows1 = im1.shape[0]
	rows2 = im2.shape[0]
	if rows1 < rows2:
		im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
	elif rows1 > rows2:
		im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
	# if none of these cases they are equal, no filling needed.

	return concatenate((im1,im2), axis=1)



def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True): 
	""" Show a figure with both the images appended and lines joining the accepted matches
	input: im1,im2 (images as arrays), locs1,locs2 (feature locations), matchscores (as output from ’match()’),
	show_below (if images should be shown below matches). """

	im3 = appendimages(im1,im2) 
	#if show_below:
	im3 = vstack((im3,im3))
	
	imshow(im3)

	cols1 = im1.shape[1]
	for i,m in enumerate(matchscores):
		if m > 0: 
			plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]],'r')

	axis('off')


# RANSAC for finding the best row offset value and the best column offset value 

def RANSAC(matches, coord_list1,coord_list2,match_dist=1.6):
    d2 = match_dist**2
    offsets = np.zeros((len(matches),2))
    for i in range(len(matches)):
        index2 = matches[i]
        offsets[i,0] = coord_list1[index2][0] - coord_list2[index2][0]
        offsets[i,1] = coord_list1[index2][1] - coord_list2[index2][1]
        
    best_match_count = -1
    best_row_offset, best_col_offset = 1e6,1e6
    for i in range(len(offsets)):
        match_count = 1.0
        offi0 = offsets[i,0]
        offi1 = offsets[i,1]
        if((offi0 - best_row_offset)**2 + (offi1 - best_col_offset)**2 ) >= d2:
            sum_row_offsets,sum_col_offsets = offi0,offi1
            for j in range(len(matches)):
                if j != 1:
                    offj0 = offsets[j,0]
                    offj1 = offsets[j,1]
                    if ((offi0 - offj0)**2 + (offi1 - offj1)**2 )< d2:
                        sum_row_offsets += offj0
                        sum_col_offsets += offj1
                        match_count += 1.0
            if match_count != best_match_count:
                best_row_offset = sum_row_offsets / match_count
                best_col_offset = sum_col_offsets / match_count
                best_match_count = match_count
    print "Exhaustive RANSAC best match count: %f" % best_match_count
    return best_row_offset, best_col_offset, best_match_count


# Connecting the 2 images based on the offset values returned
def connect(im_1, im_2, offset_row, offset_col, choice):
    x1, y1 = im_1.shape
    x2, y2 = im_2.shape
    
    
    """a1 = int(x1+offset_row)
    a2 = int(y2+offset_col)

    canvas = np.zeros((a1, a2))
    canvas = Image.fromarray(canvas)
    im1 = Image.fromarray(im_1)
    im2 = Image.fromarray(im_2)"""

    #For the arch use this to see a well aligned image#
    if choice == 1:
        
        a1 = int(x1+offset_row)
        a2 = int(y2+offset_col)

        canvas = np.zeros((a1, a2))
        canvas = Image.fromarray(canvas)
        im1 = Image.fromarray(im_1)
        im2 = Image.fromarray(im_2)
        
        canvas.paste(im1,(0,280))
        canvas.paste(im2,(145,0))
       
    if choice == 2:
        
        
         a1 = int(x1+offset_row)
         a2 = int(y2+offset_col)

         canvas = np.zeros((a1, a2))
         canvas = Image.fromarray(canvas)
         im1 = Image.fromarray(im_1)
         im2 = Image.fromarray(im_2)
         canvas.paste(im1,(0,220))    #-50
         canvas.paste(im2,(290,0))      #240
    
    # For the balloon image use this to see a well aligned image
    """canvas.paste(im1,(-50,220))
    canvas.paste(im2,(240,0)) 
    """
    
    """canvas.paste(im1,(0,int(offset_row)))
    canvas.paste(im2,(int(offset_col),0)) 
    """

    imshow(canvas)
    show()


def get_user_input():
    print "\nTo analyse the following pictures, please enter it's number and press enter: \n"
    print "\t\t(1) arch.png\n\
           \t(2) balloon.png\n\
           \t(3.) EXIT\n"
    choice = input()
    return choice
    


choice = get_user_input()
 

if choice == 1:
    im1 = np.array(Image.open('arch1.png').convert('L'))
    im2 = np.array(Image.open('arch2.png').convert('L'))
    #choice = get_user_input()
if choice == 2:
    im1 = np.array(Image.open('balloon1.png').convert('L'))
    im2 = np.array(Image.open('balloon2.png').convert('L'))
    #choice = get_user_input()
#elif choice == 3:
   
#else:
#    print "the number ", choice, " is not a valid number\n"
#    print "Please Enter a Valid number:\n"
#    choice = input()
# 
#    







harris_im1 = compute_harris_response(im1, 5)
harris_im2 = compute_harris_response(im2, 5)

# harris points 
filtered_coords_im1=get_harris_points(harris_im1,min_dist=10,threshold=0.1)
filtered_coords_im2=get_harris_points(harris_im2,min_dist=10,threshold=0.1)
wid=5

# descriptor for image 1 and image 2
d1 = get_descriptors(im1, filtered_coords_im1,wid)
d2 = get_descriptors(im2, filtered_coords_im2,wid)


# matching the corners in both the images and displaying 
matches = match_twosided(d1,d2)
plot_harris_points(im1, filtered_coords_im1)
plot_harris_points(im2, filtered_coords_im2)

print 'starting matching'
figure()
gray() 
#plot_matches(im1,im2,filtered_coords_im1,filtered_coords_im2,matches) 
#show()

best_row_offset, best_col_offset, best_match_count = RANSAC(matches,d1,d2)

print RANSAC(matches,d1,d2)

connect(im1,im2,best_row_offset,best_col_offset, choice)




"""x, y = im1.shape;

x = x + best_col_offset
y = y + best_col_offset

image = np.zeros((x, y))
image = Image.fromarray(image)
imA = Image.fromarray(im1)
imB = Image.fromarray(im2)


image.paste(imA,(0,280))
image.paste(imB,(145,0))
"""

gray()

#imshow(image)


show()

figure()
gray()
plot_matches(im1,im2,filtered_coords_im1,filtered_coords_im2,matches)
show()




                
