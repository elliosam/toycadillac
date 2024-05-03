import matplotlib.pyplot as plt
import cv2 as cv
from skimage import filters
from scipy import ndimage as ndi
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC  
from sklearn.inspection import DecisionBoundaryDisplay




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
    # getting the the pixels values for each spot
    while j<den:
         xx=cordss[:,j,0]
         yy=cordss[:,j,1]
         sss=xx.tolist()
         yyy=yy.tolist()
         pixel=blurred[sss,yyy]
         
         # separating each pixel into its r,g,b components
         red.append(pixel[:,0])
         green.append(pixel[:,1])
         blue.append(pixel[:,2])
         j+=1
         #each value is some fraction n/255
    # converting to array     
    r=np.array(red)
    E_r=np.sum(r)/den
    values_r, counts_r = np.unique(r, return_counts=True)
    p_r=counts_r/den
    inside_r=values_r-E_r
    inside_r=np.absolute(inside_r)
    tosum_r=np.multiply(inside_r,p_r)
    var_r=np.sum(tosum_r)
    b=np.array(blue)
    g=np.array(green)
    #Expected Values for r,g,b (total green value in space / number of pixels in space)
    
    E_b=np.sum(b)/den
    E_g=np.sum(g)/den
   
    #separates every every value of r, g, or b found in the image and the total amount of each
    
    values_b, counts_b = np.unique(b, return_counts=True)
    values_g, counts_g = np.unique(g, return_counts=True)
    #probability of each pixel value occuring
    
    
    p_b=counts_b/den
    p_g=counts_g/den
    # sum of counts_g = number of pixels in space 
    #probability of getting a given value
    inside_r=values_r-E_r
    inside_b=values_b-E_b
    inside_g=values_g-E_g
    # diviation of each value from mean
    inside_r=np.absolute(inside_r)
    inside_g=np.absolute(inside_g)
    inside_b=np.absolute(inside_b)
    # instead of squaring the values I  used an absolute value to get nicer results 
    # squaring values punsishes large deviation too much
    tosum_r=np.multiply(inside_r,p_r)
    tosum_b=np.multiply(inside_b,p_b)
    tosum_g=np.multiply(inside_g,p_g)
    #multiplitying each values diviation from mean by its probability of occuring
    var_r=np.sum(tosum_r)
    var_b=np.sum (tosum_b)
    var_g=np.sum(tosum_g)
    
    # final value
    var=var_r+var_b+var_g
    #please leave this d (has personal sentament)
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

 #change image   
bg = cv.imread('image_path.jpg')
print(shapes)
dics=[]
i=0
while (i < len(shapes)):
    # pre-processing
    img33 = imgg = np.zeros((bg.shape[0], bg.shape[1], 3), dtype = np.uint8)
    blurred = filters.gaussian(bg, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
    
    currentPts = np.array([[shapes[i]['x1'], shapes[i]['y1']], [shapes[i]['x2'], shapes[i]['y2']], [shapes[i]['x3'], shapes[i]['y3']], [shapes[i]['x4'], shapes[i]['y4']]])
   #outline space circle
    cv.circle(bg, ((shapes[i]['x1'],shapes[i]['y1'])), 2, (0,0,255), 2, cv.FILLED)
    cv.circle(bg, ((shapes[i]['x2'],shapes[i]['y2'])), 2, (0,0,255), 2, cv.FILLED)
    cv.circle(bg, ((shapes[i]['x3'],shapes[i]['y3'])), 2, (0,0,255), 2, cv.FILLED)
    cv.circle(bg, ((shapes[i]['x4'],shapes[i]['y4'])), 2, (0,0,255), 2, cv.FILLED)
    #findind minimum and max values of points
    ymax=np.array(max(shapes[i]['y1'],shapes[i]['y2'],shapes[i]['y3'],shapes[i]['y4']))
    ymin=np.array(min(shapes[i]['y1'],shapes[i]['y2'],shapes[i]['y3'],shapes[i]['y4']))
    xmax=np.array(max(shapes[i]['x1'],shapes[i]['x2'],shapes[i]['x3'],shapes[i]['x4']))
    xmin=np.array(min(shapes[i]['x1'],shapes[i]['x2'],shapes[i]['x3'],shapes[i]['x4']))
    #counts white pixels of space
    # please leave this d (has personal sentament)
    d=cv.fillPoly(img33, pts=[currentPts], color=(255,255, 255))
    den = int(np.sum(d == 255) / 3)
    indices = np.where(d == [255])
    cord = np.dstack((indices[0], indices[1]))
    cordss=np.unique(cord,axis = 1)
    
    # run through checking algorithms
    maxim = checkMaxima(bg,currentPts,den)
    vari = checkVariance(bg, currentPts, den)
    xmid=(xmax+xmin-20)/2
    ymid=(ymax+ymin)/2
    points=int(xmid),int(ymid)
    counter=str(i)
    cv.putText(bg,counter,(points),cv.FONT_HERSHEY_SIMPLEX, .4,(0,0,0), 2,2)
    cv.putText(bg,counter,(points),cv.FONT_HERSHEY_SIMPLEX, .4,(255,255,255), 1,2)
    maximInt = round(maxim * 100)
    variInt = round(vari * 100)

    print("SPOT " + str(i) + " MAX: " + str(maximInt) + "  VARIANCE: " + str(variInt))
    #Put red or green outline around spot 
    if shapes[i]['car']==False:
        cv.line(bg,(shapes[i]['x1'],shapes[i]['y1']),(shapes[i]['x2'],shapes[i]['y2']), (0, 255, 0), 2, cv.LINE_AA)
        cv.line(bg,(shapes[i]['x2'],shapes[i]['y2']),(shapes[i]['x3'],shapes[i]['y3']), (0, 255, 0), 2, cv.LINE_AA)
        cv.line(bg,(shapes[i]['x3'],shapes[i]['y3']),(shapes[i]['x4'],shapes[i]['y4']), (0, 255, 0), 2, cv.LINE_AA)
        cv.line(bg,(shapes[i]['x4'],shapes[i]['y4']),(shapes[i]['x1'],shapes[i]['y1']), (0, 255, 0), 2, cv.LINE_AA)
    else:
        cv.line(bg,(shapes[i]['x1'],shapes[i]['y1']),(shapes[i]['x2'],shapes[i]['y2']), (0, 0, 255), 2, cv.LINE_AA)
        cv.line(bg,(shapes[i]['x2'],shapes[i]['y2']),(shapes[i]['x3'],shapes[i]['y3']), (0, 0, 255), 2, cv.LINE_AA)
        cv.line(bg,(shapes[i]['x3'],shapes[i]['y3']),(shapes[i]['x4'],shapes[i]['y4']), (0, 0, 255), 2, cv.LINE_AA)
        cv.line(bg,(shapes[i]['x4'],shapes[i]['y4']),(shapes[i]['x1'],shapes[i]['y1']), (0, 0, 255), 2, cv.LINE_AA)    
    dics.append({ 'Max':str(maximInt), 'Variance':str(variInt), 'Is CAr?':shapes[i]['car'] })
    i += 1
    perc = 0
cv.imshow("Parking Lot", bg)
df=pd.DataFrame(dics)

# adds result to csvfile containing all run information
#df.to_csv("bigmumbojumbo.csv", mode='a',header=False,index=False)
x = pd.read_csv("bigmumbojumbo.csv")
#format csv file that into array
a = np.array(x)

y  = a[:,2] 

x = a[:,[0,1] ]

y=y.astype(int)
#SVM using polynomiaal kernal
clf = SVC(kernel='linear') 
clf.fit(x, y) 
Decision_Function=clf.decision_function(x)
_, ax = plt.subplots(figsize=(4, 3))
x_min, x_max, y_min, y_max = 0, 100, 0, 100
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
DecisionBoundaryDisplay.from_estimator(estimator=clf,X=x,ax=ax,response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],)

ax.scatter(x[:, 0], x[:, 1], c=y, s=150, edgecolors="k")
#plot
plt.xlabel("Maximal Filter")
plt.ylabel("Variance")
plt.title('Support Vector Machines Boundaries')
s=clf.score(x, y)
plt.show()
print(s)
cv.waitKey(0)
   