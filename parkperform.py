import cv2 as cv
import numpy as np
import pickle

### PARKPERFORM: this code will display the video feed with an overlay of all the parking spots
### color coded to whether or not a car is there. meant to run in parallel with parkparse.py.

cap = cv.VideoCapture('Photos\carPark.mp4')

# load shape data
try:
    with open('parkingPositions', 'rb') as f:
        shapes = pickle.load(f)
except:
    print("failed to import shape data")
    
try:
    with open('doneTF', 'rb') as f:
        done = pickle.load(f)
except:
    done = False
    print("failed to import done data")
    
def updateColorFalse(img, shapeID):
    cv.line(img, (shapes[shapeID]['x1'], shapes[shapeID]['y1']), (shapes[shapeID]['x2'], shapes[shapeID]['y2']), (0,0,255), 2, cv.LINE_AA)
    cv.line(img, (shapes[shapeID]['x2'], shapes[shapeID]['y2']), (shapes[shapeID]['x3'], shapes[shapeID]['y3']), (0,0,255), 2, cv.LINE_AA)
    cv.line(img, (shapes[shapeID]['x3'], shapes[shapeID]['y3']), (shapes[shapeID]['x4'], shapes[shapeID]['y4']), (0,0,255), 2, cv.LINE_AA)
    cv.line(img, (shapes[shapeID]['x4'], shapes[shapeID]['y4']), (shapes[shapeID]['x1'], shapes[shapeID]['y1']), (0,0,255), 2, cv.LINE_AA)

def updateColorTrue(img, shapeID):
    cv.line(img, (shapes[shapeID]['x1'], shapes[shapeID]['y1']), (shapes[shapeID]['x2'], shapes[shapeID]['y2']), (0,255,0), 2, cv.LINE_AA)
    cv.line(img, (shapes[shapeID]['x2'], shapes[shapeID]['y2']), (shapes[shapeID]['x3'], shapes[shapeID]['y3']), (0,255,0), 2, cv.LINE_AA)
    cv.line(img, (shapes[shapeID]['x3'], shapes[shapeID]['y3']), (shapes[shapeID]['x4'], shapes[shapeID]['y4']), (0,255,0), 2, cv.LINE_AA)
    cv.line(img, (shapes[shapeID]['x4'], shapes[shapeID]['y4']), (shapes[shapeID]['x1'], shapes[shapeID]['y1']), (0,255,0), 2, cv.LINE_AA)

# Check if camera opened successfully 
if (cap.isOpened() == False): 
    print("Error opening video file") 

count = 0
parkTF = []

try:
    with open('isCar', 'rb') as f:
        parkTF = pickle.load(f)
except:
    if (count < len(shapes)):
        parkTF.append(False)
        count += 1

# Read until video is completed 
while(cap.isOpened()): 
      
    # Capture frame-by-frame 
    ret, frame = cap.read() 
    if ret == True: 
        # load car data
        
        with open('isCar', 'rb') as f:
            parkTF = pickle.load(f)
            print(parkTF)
       
            #print("car data could not be loaded")

        
        i = 0
        print(str(shapes[i]["car"]))
        print(str(parkTF[i]))
        while(i < len(shapes)):
            if (parkTF[i] == True):
                updateColorTrue(frame, i)
            elif (parkTF[i] == False):
                updateColorFalse(frame, i)
            else:
                break
            i += 1
        
        # display frame
        cv.imshow('Frame', frame) 
        
        # grab the done boolean
        try:
            with open('doneTF', 'rb') as f:
                done = pickle.load(f)
        except:
            done = False
            print("failed to import done value")
        
        # if the calculations are done for a frame, send a new frame over
        if (done == True):
            try:
                with open('currentFrame', 'wb') as f:
                    pickle.dump(frame, f)
            except:
                print("could not dump current frame")
                
        # Press Q on keyboard to exit 
        if cv.waitKey(25) & 0xFF == ord('q'): 
            break
    
# Break the loop 
    else: 
        break
  
# When everything done, release 
# the video capture object 
cap.release() 
  
# Closes all the frames 
cv.destroyAllWindows() 