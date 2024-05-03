import matplotlib.pyplot as plt
import cv2 as cv
from skimage.feature import hog, peak_local_max
from skimage import data, exposure,filters,io
from scipy import ndimage as ndi
import numpy as np
import pickle

sigma = 3
red = []
blue = []
green = []
parkTF = []

# import shape data from parkpick
try:
    with open('parkingPositions', 'rb') as f:
        shapes = pickle.load(f)
except:
    print("shapes could not be loaded")


def checkMaxima(img, points, perc):
    # image preprocessing
    im = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    im = (255-im)
    
    image_max = ndi.maximum_filter(im, size = 20, mode = 'constant')
    
    #cv.imshow('image_max', image_max)
    
    # create black background and add a white polygon
    blackBG = np.zeros(shape = (im.shape[:2]), dtype = np.uint8)
    cv.fillPoly(blackBG, pts = [points], color = (255,255,255))
    
    # find x and y points
    xp = [points[0][0], points[1][0], points[2][0], points[3][0]]
    yp = [points[0][1], points[1][1], points[2][1], points[3][1]]
    
    # crop the polygon
    blackBG = blackBG[min(yp) : max(yp), min(xp) : max(xp)]
    image_max_crp = image_max[min(yp) : max(yp), min(xp) : max(xp)]
    
    # mask the maxima filter
    maximaMask = cv.bitwise_and(image_max_crp, image_max_crp, mask = blackBG)
    
    # add some thresholding
    ret, maximaMask = cv.threshold(maximaMask, 225, 255, cv.THRESH_BINARY)
    #cv.imshow('ligmaal;esd', maximaMask)
    
    
    # find the percentage of 'on' pixels
    percOn = cv.countNonZero(maximaMask)
    #print('percent on from maxima mask: ' + str(percOn))
    
    # return the ratio of percOn and the percentage we pass through the function
    ratio = percOn / perc
    #print('ratio: ' + str(ratio))
    
    # if (ratio >= 0.7):
    #     print("maxima gods have deemed car with ratio: " + str(ratio))
    # else:
    #     print("maxima gods have deemed not car with ratio: " + str(ratio))
    
    return ratio

def checkVariance(img, points, den):
    j = 0
    while j<den:
         xx=cordss[:,j,0]
         yy=cordss[:,j,1]
         sss=xx.tolist()
         yyy=yy.tolist()
         pixel=blurred[sss,yyy]
         red.append(pixel[:,0])
         green.append(pixel[:,1])
         blue.append(pixel[:,2])
         j+=1
         
    r=np.array(red)
    b=np.array(blue)
    g=np.array(green)
    
    E_r=np.sum(r)/den
    E_b=np.sum(b)/den
    E_g=np.sum(g)/den
    
    values_r, counts_r = np.unique(r, return_counts=True)
    values_b, counts_b = np.unique(b, return_counts=True)
    values_g, counts_g = np.unique(g, return_counts=True)
    
    p_r=counts_r/den
    p_b=counts_b/den
    p_g=counts_g/den
    
    inside_r=values_r-E_r
    inside_b=values_b-E_b
    inside_g=values_g-E_g
    
    inside_r=np.absolute(inside_r)
    inside_g=np.absolute(inside_g)
    inside_b=np.absolute(inside_b)
    
    tosum_r=np.multiply(inside_r,p_r)
    tosum_b=np.multiply(inside_b,p_b)
    tosum_g=np.multiply(inside_g,p_g)
    
    var_r=np.sum(tosum_r)
    var_b=np.sum (tosum_b)
    var_g=np.sum(tosum_g)
    
    # final value
    var=var_r+var_b+var_g
    d=cv.fillPoly(img33, pts=[currentPts], color=(0,0, 0))
    red.clear()
    green.clear()
    blue.clear()
    return var

# set parkTF array to correct size
count = 0
while (count < len(shapes)):
    parkTF.append(shapes[count]["car"])
    count += 1
    
bg = cv.imread('Photos/carParkImg.png')

while True:
    i = 0
    
    # load current frame from parkperform.py
    try:
        with open('currentFrame', 'rb') as f:
            bg = pickle.load(f)
    except:
        print("could not load frame from pickle")
    
    # reset done var to not get new frame until processing finishes
    done = False
    
    while (i < len(shapes)):
        # pre-processing
        img33 = imgg = np.zeros((bg.shape[0], bg.shape[1], 3), dtype = np.uint8)
        blurred = filters.gaussian(bg, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
        
        currentPts = np.array([[shapes[i]['x1'], shapes[i]['y1']], [shapes[i]['x2'], shapes[i]['y2']], [shapes[i]['x3'], shapes[i]['y3']], [shapes[i]['x4'], shapes[i]['y4']]])
    
        d = cv.fillPoly(img33, pts=[currentPts], color=(255,255, 255))
        den = int(np.sum(d == 255) / 3)
        indices = np.where(d == [255])
        cord = np.dstack((indices[0], indices[1]))
        cordss=np.unique(cord,axis = 1)

        # run through checking algorithms
        maxim = checkMaxima(bg,currentPts,den)
        vari = checkVariance(bg, currentPts, den)
        
        maximInt = round(maxim * 100)
        variInt = round(vari * 100)

        print("SPOT " + str(i) + " MAX: " + str(maximInt) + "  VARIANCE: " + str(variInt))
            
        if ((variInt >= 30) & (maximInt >= 40)):
            shapes[i].update({'car': True})
            parkTF[i] = True
        else:
            shapes[i].update({'car': False})
            parkTF[i] = False
        i += 1
        perc = 0
        
    done = True
    try:
        with open('isCar', 'wb') as f:
            pickle.dump(parkTF, f)
            print("I DID IT")
    except:
        print("could not pickle shapes values")
    
    try:
        with open('doneTF', 'wb') as f:
            pickle.dump(done, f)
    except:
        print("could not pickle done variable")