import cv2 as cv
import numpy as np
import pickle
xpoints = []
ypoints = []

try:
    with open('parkingPositions', 'rb') as f:
        shapes = pickle.load(f)#Load the data from pickled parkingPositions file into the shapes array
except:
    shapes = []
    
def point_inside_polygon(x,y,poly):# x and y are the points of the click, poly is a list of the vertices;
    inside =False#Default to click being outside

    p1x,p1y = poly[0]#Set, p1x, p1y to the first set of x,y points
    i=1
    while i<5:#Goes through the the 4 points stored in poly array;
        #Compares poly[0] and poly[1], poly[1] and poly[2], poly[2] and poly[3], and poly[3] and poly[0]
        #If a point is inside of a polygon, if you then create a line going to the right of the point,
        #you will only find and odd number of intersections with the boundaries of the polygon, but only if it's inside
        p2x,p2y = poly[i % 4] #[i % 4] saves the remainder of i divided by 4 so for i=1,2,3,4, [i % 4]=1,2,3,0
        if ((y > min(p1y,p2y)) & (y <= max(p1y,p2y))):#Verify that y is between the max and minimum of 2 of the points
            if x <= max(p1x,p2x):#Verify that x is either less than or equal to the max x
                if p1y != p2y:#if the y's from poly aren't equal
                    xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x#X intersection with boundary of polygon
                if p1x == p2x or x <= xinters:
                    inside = not inside#Click is inside                      
                    
        p1x,p1y = p2x,p2y#Set the 2nd point to the first point before going through another loop
        i=i+1
    return inside

def point_plot():#Plots the x and y points of the shape currently being drawn
    if len(xpoints) >= 1:#Place 1st dot
        cv.circle(lot, (xpoints[0],ypoints[0]), 2, (0,215,255), 2, cv.FILLED)#places dot
    if len(xpoints) >= 2:#Place 2nd dot and line connecting 1st and 2nd points
        cv.circle(lot, (xpoints[1],ypoints[1]), 2, (0,215,255), 2, cv.FILLED)
        cv.line(lot, (xpoints[0], ypoints[0]), (xpoints[1], ypoints[1]), (0,0,128), 2, cv.LINE_AA)#places line            
    if len(xpoints) >= 3:#Place 3nd dot and line connecting 2nd and 3rd points
        cv.circle(lot, (xpoints[2],ypoints[2]), 2, (0,215,255), 2, cv.FILLED)
        cv.line(lot, (xpoints[1], ypoints[1]), (xpoints[2], ypoints[2]), (0,0,128), 2, cv.LINE_AA)
    if len(xpoints) == 4:#Place 4th dot and lines connecting 3rd and 4th points and 1st and 4th points
        cv.circle(lot, (xpoints[3],ypoints[3]), 2, (0,215,255), 2, cv.FILLED)
        cv.line(lot, (xpoints[2], ypoints[2]), (xpoints[3], ypoints[3]), (0,0,128), 2, cv.LINE_AA)
        cv.line(lot, (xpoints[3], ypoints[3]), (xpoints[0], ypoints[0]), (0,0,128), 2, cv.LINE_AA)          

def shape_plot():#Plots the shapes we've already completed and stored in the shapes array
    for pos in shapes:#Run through the shapes array and assign the values to pos
        cv.line(lot, (pos['x1'], pos['y1']), (pos['x2'], pos['y2']), (0,0,128), 2, cv.LINE_AA)#Plot Line between points 1 and 2
        cv.line(lot, (pos['x2'], pos['y2']), (pos['x3'], pos['y3']), (0,0,128), 2, cv.LINE_AA)#Plot Line between points 2 and 3
        cv.line(lot, (pos['x3'], pos['y3']), (pos['x4'], pos['y4']), (0,0,128), 2, cv.LINE_AA)#Plot Line between points 3 and 4
        cv.line(lot, (pos['x4'], pos['y4']), (pos['x1'], pos['y1']), (0,0,128), 2, cv.LINE_AA)#Plot Line between points 4 and 1     

def mouseClick(events,x,y,flags,param):#Looks for a mouseclick
    if ((len(xpoints) <= 3)):#User won't be able do anything with click if there are 4 dots on the screen; Must pass through enteroccupancy to get back in
        if (events ==cv.EVENT_LBUTTONDOWN):#if left click
            xpoints.append(x)#Store the x of the click into an array
            ypoints.append(y)#Store the y of the click into an array
            print("X-points:",xpoints)
            print("Y-points:",ypoints)

        elif events ==cv.EVENT_RBUTTONDOWN:#if right click
            prioramount=len(shapes)
            postamount=len(shapes)
            for i, pos in enumerate(shapes):# Looks through shapes array
                poly=[[pos['x1'],pos['y1']],[pos['x2'],pos['y2']],[pos['x3'],pos['y3']],[pos['x4'],pos['y4']]]#Create a poly array to be used in point_inside_polygon
                result=point_inside_polygon(x,y,poly)
                if result==True:#Right click is inside of a quadrilateral
                    shapes.pop(i)#The quadrilateral is removed from shapes array
                    postamount=len(shapes)#change post amount 
                    if len(shapes) != 1:
                        print("Parking spot deleted.",len(shapes),"spots remain.")
                    else:
                        print("Parking spot deleted. 1 spot remains.")
            if (prioramount==postamount)&(len(xpoints)!=0):#If no shape was deleted(postamount wasn't changed) and the shape array isn't empty, delete last dot.
                xpoints.pop(len(xpoints)-1)#Delete most recent dot placed
                ypoints.pop(len(ypoints)-1)#Delete most recent dot placed           

        elif events == cv.EVENT_RBUTTONDBLCLK:#if double right click
            p=0
            while p!=1:   
                print("Delete all shapes, dots and lines? Type 'y' for yes or 'n' for no.")
                keycode=cv.waitKeyEx(0)#wait for input
                if keycode==121:#Number for y key
                    p=1
                    print("Clearing all spots.")
                    shapes.clear()#Reset the shapes array
                    xpoints.clear()#Reset the xpoints array
                    ypoints.clear()#Reset the ypoints array
                if keycode==110:#Number for n key
                    p=1
                    print("Clearing cancelled.")
                if ((keycode!=121) & (keycode!=110)):#If input in neither 'n' or 'y', 
                    print("Invalid input. Try again.")
                cv.imshow('Lot Feed', lot)

        
        with open('parkingPositions', 'wb') as f:
            pickle.dump(shapes, f)#Store the current data in shapes into the parkingPositions file, so the spot data can be used with parkparkse and parkperform

def enteroccupancy():#Used to determine if the spot displays a car or not; Not really utilized in our code, but might be useful in collecting data for a dataset in the future
    b=0#When 0, no valid input has been given; once 1, shapes array will be appended and this module will be exited
    while len(xpoints)==4:#User won't be able to leave until the shape is put into the shapes array
        while b != 1:
            print("Is it a car? Type 'y' for yes, or 'n' for no.")
            keycode=cv.waitKeyEx(0)#wait for input
            if keycode==121:#Number for y key
                TruFals=True
                b=1
                print("Shape declared as containing a car.")
            if keycode==110:#Number for n key
                TruFals=False
                b=1
                print("Shape declared as not containing a car.")
            if ((keycode!=121) & (keycode!=110)):#If input in neither 'n' or 'y', 
                print("Invalid input. Try again.")
            if b==1:
                shapes.append({#Stores the information of all the quadrilaterals the user draws
                    "x1": xpoints[0],
                    "y1": ypoints[0],
                    "x2": xpoints[1],
                    "y2": ypoints[1],
                    "x3": xpoints[2],
                    "y3": ypoints[2],
                    "x4": xpoints[3],
                    "y4": ypoints[3],
                    "car": TruFals
                })
                xpoints.clear()
                ypoints.clear()
                cv.imshow('Lot Feed', lot)
                with open('parkingPositions', 'wb') as f:
                    pickle.dump(shapes, f)#Update the pickle file with the updated info from the shapes array  

while True:
    lot=cv.imread('Photos/carParkImg.png')#Read in the image
    point_plot()#Plots the Lot Feed associated with each mouseclick
    shape_plot()#Plots the quadrilaterals listed in the shapes array
    cv.imshow('Lot Feed',lot)#Display the parking lot feed image    
    cv.setMouseCallback('Lot Feed',mouseClick)#Look for mouse clicks on the 'Lot Feed' window
    enteroccupancy()#Puts user in a loop until they provide a proper input for if there is a car
    if cv.waitKey(1) & 0xFF == ord('q'):#Press q to exit program
        break
